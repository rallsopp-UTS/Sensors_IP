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
        self.ff_data = mujoco.MjData(self.model)

        #create camera (refer to as renderer)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer_watcher = mujoco.Renderer(self.model, height=480, width=640)

        self.arm_joints = {"shoulder_pan_joint" : 0, "shoulder_lift_joint": 1, "elbow_joint": 2, "wrist_1_joint": 3, "wrist_2_joint": 4, "wrist_3_joint": 5}
        self.arm_ids = list(self.arm_joints.values())
        self.grip_id = self.model.actuator("gripper/fingers_actuator").id
        print("Arm Ids:", self.arm_ids)
        print("Grip Id:", self.grip_id)

        #Calibration of Camera
        self.K, self.dist = self.CalibrateCamera(dir="calib_2")
        self.cam_id = self.model.camera("RGB-D").id

        #set to home state (roughly looking at cube/tag)
        for d in [self.data, self.est_data, self.ff_data]:
            mujoco.mj_resetDataKeyframe(self.model, d, 0)
            mujoco.mj_forward(self.model, d)

        self.A, self.B, self.C, self.ctrb, self.obsv, self.Kp, self.G0, self.u0, self.q0 = self.StateSpaceModel()
        self.Ts = self.model.opt.timestep
        self.A_d, self.B_d, self.C_d = self.Discretisation(self.Ts)

        self.sig_q = 0.002 # encoder noise, rad
        self.rng = np.random.default_rng(0)
        #self.Qd = np.diag([1e-4**2]*6 + [5e-2**2 * 27.5]*6) * self.Ts * 3
        w_pos = np.array([4.55e-7, 2.59e-6, 1.44e-6, 2.41e-7, 3.07e-7, 6.63e-8])
        w_vel = np.array([3.70e-4, 2.28e-3, 1.13e-3, 1.26e-4, 2.44e-4, 1.15e-4])
        self.Qd = np.diag(np.concatenate([w_pos, w_vel]) ** 2) * 120
        self.Rn = np.eye(6) * self.sig_q**2
        self.xh = np.zeros(12)
        self.P = np.eye(12) * 1e-2
        self.q_hat = self.q0.copy()
        
        #LQR Stuff
        self.K_lqr = self.LQR()
        
        #PBVS Stuff
        self.Rx_pi = np.diag([1.0, -1.0, -1.0])
        self.c_starTm = SE3.Trans(0, 0, 0.15) @ SE3.Rx(pi) @ SE3.Rz(pi/2) @ SE3.Trans(0.01, 0, 0)

    def TakePhoto(self): 
        """
        Takes two photos, one in RGB colour from the arm, one from the full scene view camera. Returns both
        """
        
        self.renderer.update_scene(self.data, camera="RGB-D")
        rgb = self.renderer.render()
        self.renderer_watcher.update_scene(self.data, camera="fixed_cam")
        full_view = self.renderer_watcher.render()
        return rgb, full_view

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

    def CalibrateCamera(self, dir): 
        """
        Use captured checkerboard images to calibrate camera matrix. Return Camera intrinsic matrix (K) and distortion (1x6 array)
        """

        calib_dir = os.path.abspath(dir)
        images = FileCollection(os.path.join(calib_dir, "img_*.png"))
        print(f"{len(images)} images found") 

        K, dist, frames = CentralCamera.images2C(images, squaresize=0.01)
        print("Calibrated")
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

    def LQR(self):
        q_max  = np.deg2rad([2]*6)   # position error rad
        qd_max = np.deg2rad([20]*6)   # velocity error rad/s
        u_max  = np.array([0.1]*6) 
        
        Q_lqr = np.diag(np.concatenate([1 / q_max**2, 1 / qd_max**2]))
        R_lqr = np.diag(1 / u_max**2)
        
        K_lqr, _, E_lqr = ct.dlqr(self.A_d, self.B_d, Q_lqr, R_lqr)
        
        return K_lqr

    def ControlLQR(self, x_hat, q_ref):
        e_q = x_hat[:6] - q_ref
        e_qd = x_hat[6:]
        e = np.hstack([e_q, e_qd])
        
        u_total = self.Get_U_Feedforward(q_ref) - self.K_lqr @ e # minus u feedback
        
        return u_total 
    
    def Get_U_Feedforward(self, q_ref):

        self.ff_data.qpos[:6] = q_ref #test estimated location
        self.ff_data.qvel[:] = 0 #at location with no velocity
        
        mujoco.mj_forward(self.model, self.ff_data) #calculate scene
        
        gravity_torque = self.ff_data.qfrc_bias[:6] #calculate the bias force which keeps at location q_ref
        
        return q_ref + gravity_torque/np.diag(self.Kp) # return the u feedforward
    
    def DisplayImage(self, images):
        """
        Takes a list/tuple of image/s and displays them on the same plot
        """
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

    def DetectMarker(self, rgb, show=False):
        """
        Detects an ArUco marker in an RGB img and provides the Pose from camera to marker cTm. show=True to display the image/s
        """
        im_rgb = Image(rgb)
        if show:
            self.DisplayImage(im_rgb)
        fids = im_rgb.fiducial(dict="4x4_50", K=self.K_cam, side=0.075) #detection of the marker
        if len(fids) == 0 or fids[0].pose == None:
            print("Marker not Detected")
            return None
        print("Marker Detected")
        return fids[0].pose

    def PBVS(self, rgb, q_hat, show=False, Lambda=0.1, mu=0.02):
        d = self.est_data
        
        d.qpos[self.arm_ids] = q_hat
        mujoco.mj_kinematics(self.model, d)
        
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_camlight(self.model, d)
        
        cTm = self.DetectMarker(rgb, show)
        if cTm is None:
            return None, None #return realy if marker is not detected
        
        cTc_star = cTm @ self.c_starTm.inv()
        theta, u = cTc_star.angvec()
        theta_u = theta * np.asarray(u)
        err = np.hstack([cTc_star.t, theta_u]).T
        
        v_c = Lambda * err 
        
        wRc = d.cam_xmat[self.rgb_cam_id].reshape(3, 3) 
        wRc = wRc @ self.Rx_pi
        wtc = d.cam_xpos[self.rgb_cam_id].copy()
        
        wRe = d.site("attachment_site").xmat.reshape(3, 3) 
        wRe = wRe @ self.Rx_pi
        wte = d.site("attachment_site").xpos.copy()
        
        eRc = wRe.T @ wRc
        etc = wRe.T @ (wte - wtc) #multiply by wRe.T to make the same frame
        
        eVc = np.block([[eRc, skew(etc) @ eRc],
                        [np.zeros((3,3)), eRc]])
        
        v_e = eVc @ v_c #velocity end effector
        
        site_id = self.model.site("attachment_site").id
        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, d, Jp, Jr, site_id)
        
        J_world = np.vstack([Jp, Jr])[:, self.arm_ids] #slices to the dofs of the robot 0-5 instead of the entire scene with the freejoint cube
        eXw = np.block([[wRe.T, np.zeros((3, 3))], #eXw = world to end effector transformation
                        [np.zeros((3, 3)), wRe.T]])
        
        J_ee = eXw @ J_world
        J_pinv_dls = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + mu**2 * np.eye(6))
        qdot = J_pinv_dls @ v_e
        print("qdot:", qdot)
        
        return qdot, err

    def RunPick(self, T=10.0):
        model = self.model
        data = self.data
        arm = 6
        dt = model.opt.timestep
        n_steps = int(T / dt)
        cam_every = 50 # TODO Ryan Camera rate in steps

        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        self.xh[:] = 0
        self.P = np.eye(12) * 1e-2
        u_prev = self.u0.copy()
        x_hat = np.hstack([self.q0, np.zeros(arm)])              # estimate of [q, qdot]
        wTm = None
        q_ref = self.q0.copy()
        q_seen = self.q0.copy()
        visual_err = np.zeros(6)
        log = {"t": [], "x_true": [], 'x_hat': [], "u": [], "sigma": [], "P": [], "nis": [], "PBVS visual error": [], "q_meas": []}


        for k in range(n_steps):
            #1. SENSING (Part B)
            q_meas = self.ReadEncoders() #simulated encoder, the only place plant state is read for the loop.
            q_hat, qd_hat = self.KFStep(u_prev, q_meas)
            x_hat = np.hstack([q_hat, qd_hat]) 
            
            if k % cam_every == 0:
                rgb, depth = self.TakePhoto()
                cv2.imshow("wrist cam", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); cv2.imshow("scene view", cv2.cvtColor(full_view, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                qd_ref, visual_err = self.PBVS(rgb, q_hat, Lambda=Lambda)
                
                if qd_ref is None:
                    visual_err = np.full(6, np.nan)
                    lost = True
                else:
                    lost = False
                    q_seen = q_hat.copy()
                
                
                print("meters", np.linalg.norm(visual_err[:3]))
                print("rads", np.linalg.norm(visual_err[3:]))
            if lost:
                qd_ref = np.clip(2.0 * (q_seen - q_ref), -0.3, 0.3)
            q_ref = q_ref + qd_ref * dt 

            # 3. CONTROL (Part D)
            # TODO (part D): PBVS outer loop into joint target, LQR inner loop uses x_hat to compute u
            u = self.ControlLQR(x_hat, q_ref)

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
            log["PBVS visual error"].append(visual_err) 
            log["q_meas"].append(q_meas)
            # TODO (Part E): also log targets true Postion
        data.ctrl[self.grip_id] = 255
        mujoco.mj_step(model, data)
        for _ in range(75):
            rgb, full_view = self.TakePhoto()
            cv2.imshow("wrist cam", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); cv2.imshow("scene view", cv2.cvtColor(full_view, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            mujoco.mj_step(model, data)
        
        cv2.destroyAllWindows()
        return log

    def ReplayKF(self, log, q_scale=1.0, r_scale=1.0, s=500):
        """Re-runs the filter on the logged run with Q and R scaled. 1.0 = current setting."""
        A, B, C = self.A_d, self.B_d, self.C_d
        Qd, Rn = self.Qd * q_scale, self.Rn * r_scale
        U = np.array(log["u"]); Ym = np.array(log["q_meas"]); Xt = np.array(log["x_true"])
        xh = np.zeros(12); P = np.eye(12) * 1e-2
        E, NEES, NIS, COV = [], [], [], []
        for k in range(len(Ym)):
            u_prev = self.u0 if k == 0 else U[k - 1]
            xp = A @ xh + B @ (u_prev - self.u0)
            Pp = A @ P @ A.T + Qd
            S = C @ Pp @ C.T + Rn
            K = Pp @ C.T @ np.linalg.inv(S)
            nu = (Ym[k] - self.q0) - C @ xp
            xh = xp + K @ nu
            IKC = np.eye(12) - K @ C
            P = IKC @ Pp @ IKC.T + K @ Rn @ K.T
            if k >= s:
                e = Xt[k] - np.hstack([xh[:6] + self.q0, xh[6:]])   # truth used only for scoring
                E.append(e); NEES.append(e @ np.linalg.solve(P, e)); NIS.append(nu @ np.linalg.solve(S, nu))
                COV.append((np.abs(e) < 3 * np.sqrt(np.diag(P))).mean())
        E = np.array(E)
        return dict(rmse_pos=np.sqrt((E[:, :6]**2).mean()), rmse_vel=np.sqrt((E[:, 6:]**2).mean()),
                    nees=np.mean(NEES), nis=np.mean(NIS), cover3=np.mean(COV))

    def SweepTable(self, log, values=(0.1, 0.3, 1, 3, 10)):
        for name in ("Q", "R"):
            print(f"\n{name}-scale  RMSEpos   RMSEvel    NEES    NIS   cover3   (1.0 = current setting)")
            for f in values:
                m = self.ReplayKF(log, q_scale=f if name == "Q" else 1.0, r_scale=f if name == "R" else 1.0)
                print(f"{f:7.1f}  {m['rmse_pos']:9.5f} {m['rmse_vel']:9.4f} {m['nees']:7.1f} {m['nis']:6.1f} {m['cover3']:8.3f}")

    
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

        input("Run Over. Input to close")
        
        return results
        
    def EstimateQ(self, log, s=500):
        """Measures how wrong the linear model is, step by step, along a logged run."""
        A, B = self.A_d, self.B_d
        U = np.array(log["u"]); Xt = np.array(log["x_true"])
        Xd = Xt.copy(); Xd[:, :6] -= self.q0                 # deviation from keyframe
        W = np.array([Xd[k + 1] - (A @ Xd[k] + B @ (U[k] - self.u0))
                      for k in range(s, len(Xd) - 1)])
        rms = np.sqrt((W ** 2).mean(0))
        print("model error per step (RMS)  pos:", rms[:6], "\n vel:", rms[6:])
        print("current Qd (sqrt of diag)   pos:", np.sqrt(np.diag(self.Qd)[:6]), "\n vel:", np.sqrt(np.diag(self.Qd)[6:]))
        return np.diag(rms ** 2)

if __name__ == '__main__':
    run = IP()
    
    log = run.RunPick(T=30.0, Lambda=0.3)
    run.EvaluateEstimator(log)
    #run.SweepTable(log) # this is if you need tables
    #run.SweepTable(log, values=(1, 1.25, 1.5, 2, 3)) # this was to find q since I was close, it was up to 100 to narrow down the location
    
    run.renderer.close()
    run.renderer_watcher.close()
