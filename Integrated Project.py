import mujoco, mujoco.viewer
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from machinevisiontoolbox import FileCollection, CentralCamera, Image
from spatialmath import SE3
from spatialmath.base import *
from math import pi
import control as ct

class IP:
    def __init__(self):
        print("I have started!")
       #Location of ur3e_scene.xml
        self.scene_path = os.path.dirname(os.path.abspath(__file__))

        #create model from xml
        self.model = mujoco.MjModel.from_xml_path(os.path.join(self.scene_path, "ur3e_scene.xml"))

        #holds all data from model
        self.data = mujoco.MjData(self.model)
        self.est_data = mujoco.MjData(self.model)

        #dimensions of images taken by RGB-D
        self.height = 480
        self.width = 640

        #create camera (refer to as renderer)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.arm_joints = {"shoulder_pan_joint" : 0, "shoulder_lift_joint": 1, "elbow_joint": 2, "wrist_1_joint": 3, "wrist_2_joint": 4, "wrist_3_joint": 5}
        self.arm_ids = list(self.arm_joints.values())
        print("Arm Ids:", self.arm_ids)

        #Calibration of Camera
        self.K, self.dist = self.CalibrateCamera(dir="calib_2")

        #set to home state (roughly looking at cube/tag)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_resetDataKeyframe(self.model, self.est_data, 0

        #Whenever you change specific things but dont call mujoco.mj_step() make sure you use mj_forward
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.est_data)

        self.A, self.B, self.C, self.ctrb, self.obsv, self.Kp, self.G0, self.u0, self.q0 = self.StateSpaceModel()
        self.Ts = self.model.opt.timestep
        self.A_d, self.B_d, self.C_d = self.Discretisation(self.Ts)

        self.sig_q = 0.002 # encoder noise, rad
        self.rng = np.random.default_rng(0)
        self.Qd = np.diag([1e-4**2]*6 + [5e-2**2]*6) * self.Ts
        self.Rn = np.eye(6) * self.sig_q**2
        self.xh = np.zeros(12)
        self.P = np.eye(12) * 1e-2
        self.q_hat = self.q0.copy()
        
        self.c_starTm = SE3.Trans(-0.05, 0, 0.05) @ SE3.Rx(pi)

    def TakePhoto(self): #Takes two photos, one in RGB colour, one in depth. Returns both
        self.renderer.update_scene(self.data, camera="RGB-D")
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def CaptureCalibrationImages(self, n=20, out_dir="calib", seed=0, max_tries=200): #Was used to get checkerboard images for calibration
        out_dir = os.path.abspath(out_dir)
        dbg_dir = os.path.join(out_dir, "debug")
        os.makedirs(dbg_dir, exist_ok=True)
        print("saving to:", out_dir)

        rng = np.random.default_rng(seed)
        cam_id = self.model.camera("RGB-D").id
        mocap_id = self.model.body("Checkerboard").mocapid[0]
        if mocap_id < 0:
            raise RuntimeError("body 'board' is not a mocap body")

        mujoco.mj_forward(self.model, self.data)
        cam_pos = self.data.cam_xpos[cam_id].copy()
        R_cam = self.data.cam_xmat[cam_id].reshape(3, 3).copy()

        saved = 0
        for tries in range(max_tries):
            if saved >= n:
                break
            depth = rng.uniform(0.05, 0.20)
            x, y = rng.uniform(-0.02, 0.02, size=2)
            p_w = cam_pos + R_cam @ np.array([x, y, -depth])

            tilt = Rot.from_euler("xyz", rng.uniform(-50, 50, 3), degrees=True)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (R_cam @ tilt.as_matrix()).flatten())

            self.data.mocap_pos[mocap_id] = p_w
            self.data.mocap_quat[mocap_id] = quat
            mujoco.mj_forward(self.model, self.data)

            rgb, _ = self.TakePhoto()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{dbg_dir}/try_{tries:03d}.png", bgr)      # every attempt

            found, _ = cv2.findChessboardCorners(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (7, 6))
            print(f"try {tries}: found={found}")
            if found:
                cv2.imwrite(f"{out_dir}/img_{saved:02d}.png", bgr)
                saved += 1
        print(f"saved {saved} usable images out of {tries + 1} tries")

    def CalibrateCamera(self, dir): #Use captured checkerboard images to calibrate camera matrix. Return Camera intrinsic matrix (K) and distortion (1x6 array)
        calib_dir = os.path.abspath(dir)
        images = FileCollection(os.path.join(calib_dir, "img_*.png"))
        print(f"{len(images)} images found") 

        K, dist, frames = CentralCamera.images2C(images, squaresize=0.01)
        print(f"{len(frames)} of {len(images)} images used")
        print("K Matrix:", K)
        print("Distortion Factors:",dist)
        return K, dist

    def StateSpaceModel(self): #
        # Reset to the key frame and read the actuator gains
        model = self.model                                                                              # Loading the XML
        data = self.est_data                                                                            # Copy of data so self.data does not need to be touched
        arm = 6                                                                                         # number of joint arms
        mujoco.mj_resetDataKeyframe(model, data, 0)                                                     # Puts the scratch data at keyframe 0
        mujoco.mj_forward(model, data)                                                                  # compute gravity and the other derived quantities for that pose
        Kp = np.diag(model.actuator_gainprm[:, 0])                                                      # Servo stiffness per joint
        Kv = np.diag(-model.actuator_biasprm[:, 2])                                                     # Servo damping per joint

        # Find the command that holds the arm still
        data.ctrl[:] = data.qpos[:arm] + data.qfrc_bias[:arm] / np.diag(Kp)                             # command = current angle + (gravity torque / servo stiffness)
        mujoco.mj_forward(model, data)                                                                  # refresh derived quantities with new command
        self.q0 = data.qpos[:arm].copy()                                                                # save the key frame joint angles, the opperating point the model is buklt around
        self.u0 = data.ctrl[:arm].copy()                                                                # save the command that holds that pose, used later as gravity feedforward

        # Build A and B from the equations of motion
        nv = model.nv                                                                                   # number of joint velocities (6 am + 6 for the cube)
        DfDq, DfDv, DfDa = np.zeros((nv, nv)), np.zeros((nv, nv)), np.zeros((nv, nv))                   # empty arrray to be fill by Mojoco
            # DfDq = how the need force changes as joint position changes
            # DfDv = how the needed force changes as joint velocity changes
            # DfDa = How the needed force changes as joint accelerations changes
        mujoco.mjd_inverseFD(model, data, 1e-6, False, DfDq, DfDv, DfDa, None, None, None, None)        # finite invers jacobians of the invers dynamics, False = leave out actuator forces
        M0 = DfDa[:arm, :arm]                                                                           # Mass matrix of the arm at this pose
        G0 = DfDq[:arm, :arm]                                                                           # Gravity Jacobian, how gravity changes as joints move

        # Assemble A, B and the output matrix
        Minv = np.linalg.inv(M0)                                                                        # Turns Torque into acceleration
        Z = np.zeros((arm, arm))                                                                        # 6x6 block of zeros
        I = np.eye(arm)                                                                                 # 6x6 identity block
        A = np.block([[Z, I], [-Minv @ (Kp + G0), -Minv @ Kv]])                                         # 12x12 system matrix: top row says qdot is rate of change of q, bottom row is acceleration from servo spring, gravit and damping
        B = np.vstack([Z, Minv @ Kp])                                                                   # 12x6 input matrix: comandsonly affect acceleration, through the servo stiffness
        C = np.hstack([I, Z])                                                                           # 6x12 output matrix: the encoders measure the six joint angles, not the speed

        # Rank checks and return
        n = A.shape[0]                                                                                  # number of states (12)
        s = np.linalg.norm(A, 2)                                                                        # overall size of A, used to rescale time
        ctrb = ct.ctrb(A/s, B/s)                                                                        # The scaled controlability matrix
        obsv = ct.obsv(A/s, C)                                                                          # The scaled Observability matrix
        print("Controllability rank:", np.linalg.matrix_rank(ctrb), "of", n)                            # 12 of 12 means the commands can reach every state
        print("Observability rank:", np.linalg.matrix_rank(obsv), "of", n)                              # 12 of 12 means every state can be worked out by the encoders
        return A, B, C, ctrb, obsv, Kp, G0, self.u0, self.q0                                            # hand the model and perating point to the rest of the system

    def Discretisation(self, Ts):
        sysc = ct.ss(self.A, self.B, self.C, np.zeros((self.C.shape[0], self.B.shape[1])))
        sysd = ct.c2d(sysc=sysc, Ts=Ts, method="zoh")
        return sysd.A, sysd.B, sysd.C

    def ReadEncoders(self):
        return self.data.qpos[:6] + self.rng.normal(0, self.sig_q, 6)

    def KFStep(self, u_prev, q_meas):
        A, B, C = self.A_d, self.B_d, self.C_d
        u = u_prev - self.u0
        y = q_meas - self.q0
        xp = A @ self.xh + B @ u # predict
        Pp = A @ self.P @ A.T + self.Qd
        S = C @ Pp @ C.T + self.Rn
        K = Pp @ C.T @ np.linalg.inv(S)
        nu = y - C @ xp
        self.xh = xp + K @ nu # correct
        IKC = np.eye(12) - K @ C
        self.P = IKC @ Pp @ IKC.T + K @ self.Rn @ K.T
        self.nis = nu @ np.linalg.solve(S, nu)
        self.q_hat = self.xh[:6] + self.q0
        return self.q_hat, self.xh[6:]
        
    def DisplayImage(self, images): #Takes a list/tuple of image/s [im1, im2, ...] and displays them on the same plot
        if not isinstance(images, (list, tuple)):
            images = [images]
        fig, axes = plt.subplots(1, len(images), squeeze=False)
        axes = axes.flatten()
        for i, img in enumerate(images):
            im = axes[i].imshow(img, cmap="viridis")
            if img.ndim == 2:
                fig.colorbar(im,ax=axes[i], label="depth [m]", shrink=0.5)
        plt.tight_layout()
        plt.show()
            
    def SimTest(self): #runs a sim of the scene that you can fiddle with using the control tab on the right gui
        model = self.model
        data = self.data
        # Reset to the default home keyframe defined in the MJCF
        mujoco.mj_resetDataKeyframe(model, data, 0)     

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

    def DetectMarker(self, rgb, show=False): #detects an ArUco marker in frame and provides the Pose from camera to marker cTm. show=True to display the image/s
        """
        Detects an ArUco marker in an RGB img and provides the Pose from camera to marker cTm. show=True to display the image/s
        """
        im_rgb = Image(rgb)
        if show:
            self.DisplayImage(im_rgb)
        fids = im_rgb.fiducial(dict="4x4_50", K=self.K, side=0.075) #detection of the marker
        print(f'{len(fids)} marker(s) detected')
        print('id', fids[0].id)
        if fids[0].pose == None:
            return False
        return fids[0].pose

    def cTm_2_wTm(self, cTm):
        """
        Input camera to marker pose as SE3. Returns world to marker pose as SE3
        """
        d = self.est_data #Doesnt read from ground truth as long as it reads from est_data
        d.qpos[:6] = self.q_hat
        mujoco.mj_kinematics(self.model, d) #calculate forward kinematics for bodies and cameras
        mujoco.mj_camlight(self.model, d)

        R_cam_mj = d.cam_xmat[0].reshape(3, 3) #cam_xmat returns 1x9, need to reshape to 3x3 for SE3.Rt

        R_cam = SE3.Rt(R_cam_mj, [0,0,0]) @ SE3.Rx(pi) #turning mj camera along Z- and y up, to openCV along Z+ and y up by pi rotation in x

        t_cam = d.cam_xpos[0].copy() #take a snapshot of translation at that point in time

        wTc = SE3.Rt(R_cam.R, t_cam) 
        wTm = wTc @ cTm
        return wTm

    def PBVS(self, Lambda): #Unfinished
        rgb, depth = self.TakePhoto()
        cTm = self.DetectMarker(Image(rgb))
        print("cTm:", cTm)
        cTc_star = cTm @ self.c_starTm.inv()
        print("cTc_star:", cTc_star)
        theta, u = cTc_star.angvec()
        print("angle:", theta)
        print("axis:", u)
        theta_u = theta * np.asarray(u)
        print("theta_u:", theta_u)
        err = np.hstack([cTc_star.t, theta_u]).T
        print("error:", err)
        
        v_c = -Lambda * np.hstack([cTc_star.R.T @ cTc_star.t, theta_u]).T
        print("velocity camera:", v_c)

    def RunPick(self, T=10.0):
        model = self.model
        data = self.data
        arm = 6
        dt = model.opt.timestep
        n_steps = int(T / dt)
        cam_every = 50 # TODO Ryan Camera rate in steps, I just picked a random number Choose whatever even rendering camera every step

        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        self.xh[:] = 0
        self.P = np.eye(12) * 1e-2
        u_prev = self.u0.copy()
        x_hat = np.hstack([self.q0, np.zeros(arm)])              # estimate of [q, qdot]
        wTm = None                                               # latest marker pose in world frame from camera
        log = {"t": [], "x_true": [], 'x_hat': [], "u": [], "sigma": [], "P": [], "nis": []}


        for k in range(n_steps):
            #1. SENSING (Part B)
            q_meas = self.ReadEncoders() #simulated encoder, the only place plant state is read for the loop. TODO (Part B): add encoder noise
            if k % cam_every == 0:
                rgb, depth = self.TakePhoto()
                # TODO Hande No camera in view, add deapth noise
                # TODO cTm_2_wTm reads the camera pose from est_data
            
            # 2. ESTIMATION (Part C)
            # TODO (Part C) update x_hat from q_meas and the last command u, using A and C, Q and R
            _hat, qd_hat = self.KFStep(u_prev, q_meas)
            x_hat = np.hstack([q_hat, qd_hat]) 
            # 3. CONTROL (Part D)
            # TODO (part D): PBVS outer loop into joint target, LQR inner loop uses x_hat to compute u
            u = self.u0

            x_true_k = np.hstack([data.qpos[:arm], data.qvel[:arm]])
            #4. PLANT
            data.ctrl[:arm] = u
            mujoco.mj_step(model, data)
            u_prev = u

            #5. LOGGING (Part E)
            log["t"].append(data.time)
            log["x_true"].append(x_true_k)
            log["x_hat"].append(x_hat.copy())
            log["u"].append(u.copy()) 
            log["sigma"].append(np.sqrt(np.diag(self.P)))
            log["P"].append(self.P.copy())
            log["nis"].append(self.nis)
            # TODO (Part E): also log targets true Postion

    
    def EvaluateEstimator(self, log, s=500, joint=6, plot=True):
        # Scores the Kalman filter against ground truth. Truth is used ONLY here, never in the filter.
        t  = np.array(log["t"]);  Xt = np.array(log["x_true"])
        Xh = np.array(log["x_hat"]); sd = np.array(log["sigma"]); P = np.array(log["P"])

        if plot:
            j = joint # 6 = joint 1 SPEED (never measured by the encoders)
            plt.plot(t, Xt[:, j], 'k', label='true')
            plt.plot(t, Xh[:, j], 'r--', label='estimate')
            plt.fill_between(t, Xh[:, j]-3*sd[:, j], Xh[:, j]+3*sd[:, j], alpha=0.2, label='±3σ')
            plt.legend(); plt.show()

        err = Xt - Xh  # s = samples skipped for the start-up period
        nees = [e @ np.linalg.solve(Pk, e) for e, Pk in zip(err[s:], P[s:])]
        results = {
            "rmse_pos": np.sqrt((err[s:, :6]**2).mean()),
            "rmse_vel": np.sqrt((err[s:, 6:]**2).mean()),
            "nees": np.mean(nees),
            "nis": np.nanmean(log["nis"][s:]),
        }
        print("RMSE position:", results["rmse_pos"])
        print("RMSE speed   :", results["rmse_vel"])
        print("NEES (aim ~12):", results["nees"])
        print("NIS  (aim ~6) :", results["nis"])
        return results


if __name__ == '__main__':
    run = IP()
    # run.SimTest()
    # run.DisplayImage(run.TakePhoto())
    # run.CaptureCalibrationImages(out_dir="calib_2")
    log = run.RunPick(T=5.0)
    run.EvaluateEstimator(log)

    run.renderer.close()
