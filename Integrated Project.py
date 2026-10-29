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
        mujoco.mj_resetDataKeyframe(self.model, self.est_data, 0)

        #Was for testing something
        # self.data.qpos[:6] = [-0.44, -0.314, -0.157, -1.7, -0.44, -1.13]
        # self.data.ctrl[:6] = self.data.qpos[:6]

        #Whenever you change specific things but dont call mujoco.mj_step() make sure you use mj_forward
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.est_data)

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

    def DetectMarker(self, K, show=False): #detects an ArUco marker in frame and provides the Pose from camera to marker cTm. show=True to display the image/s
        rgb, depth = self.TakePhoto()
        im_rgb = Image(rgb)
        if show:
            self.DisplayImage([im_rgb, depth])
        fids = im_rgb.fiducial(dict="4x4_50", K=self.K, side=0.10) #detection of the marker
        print(f'{len(fids)} marker(s) detected')
        print('id', fids[0].id)
        print('corners (px):'); print(fids[0].corners)
        print('marker pose in the camera frame, cTm:'); print(fids[0].pose) #Pose
        return fids[0].pose

    def cTm_2_wTm(self, cTm): #Doesnt read from ground truth as long as it reads from est_data
        d = self.est_data
        for i in self.arm_ids:
            d.qpos[i] += 0.2 #debug, need real est of q for transforms
        mujoco.mj_kinematics(self.model, d) #calculate forward kinematics for bodies and cameras
        mujoco.mj_camlight(self.model, d)
        R_cam_mj = d.cam_xmat[0].reshape(3, 3) #cam_xmat returns 1x9, need to reshape to 3x3 for SE3.Rt
        R_cam = R_cam_mj @ SE3.Rx(pi) #turning mj camera along Z- and y up, to openCV along Z+ and y up by pi rotation in x
        t_cam = d.cam_xpos[0].copy() #take a snapshot of translation at that point in time
        wTc = SE3.Rt(R_cam, t_cam) 
        wTm = wTc @ cTm
        return wTm


if __name__ == '__main__':
    run = IP()
    # run.SimTest()
    # run.DisplayImage(run.TakePhoto())
    # run.CaptureCalibrationImages(out_dir="calib_2")
    run.DetectMarker(run.K)
