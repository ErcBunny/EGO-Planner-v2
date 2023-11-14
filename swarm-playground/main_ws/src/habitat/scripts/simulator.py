import rospy
import std_msgs
from sensor_msgs.msg import PointCloud2, PointField, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import math
import threading
import cv2

import numpy as np
import open3d as o3d
import habitat_sim
from habitat_sim.simulator import ObservationDict
from scipy.spatial.transform import Rotation


class OdomHandle:
    agent_id: int

    translation_world2body_mesh: np.ndarray
    rotation_world2body_mesh: Rotation
    rotation_worldframe_habitat2mesh: Rotation
    rotation_bodyframe_mesh2habitat: Rotation

    lck: threading.Lock

    def get_pose_habitat_frame(self) -> (np.ndarray, np.ndarray):
        self.lck.acquire()
        translation_world2body_habitat = self.rotation_worldframe_habitat2mesh.apply(
            self.translation_world2body_mesh
        )
        rotation_world2body_habitat = (
            self.rotation_worldframe_habitat2mesh
            * self.rotation_world2body_mesh
            * self.rotation_bodyframe_mesh2habitat
        )

        self.lck.release()
        return translation_world2body_habitat, rotation_world2body_habitat.as_quat()

    def odom_sub_cb(self, msg: Odometry):
        self.lck.acquire()
        self.translation_world2body_mesh = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        self.rotation_world2body_mesh = Rotation.from_quat(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        self.lck.release()
        self.pose_available = True

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        rospy.Subscriber(
            rospy.get_param("~odom_topic" + "_agent_" + str(agent_id)),
            Odometry,
            self.odom_sub_cb,
        )

        self.rotation_worldframe_habitat2mesh = Rotation.from_matrix(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )
        self.rotation_bodyframe_mesh2habitat = Rotation.from_matrix(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        )
        self.lck = threading.Lock()
        self.translation_world2body_mesh = np.array([0, 0, 0])
        self.rotation_world2body_mesh = Rotation.from_quat([0, 0, 0, 1])


class SyncSimulator:
    fps: int
    simulator: habitat_sim.Simulator
    agent_count: int

    global_point_cloud_points: np.ndarray
    global_point_cloud_pub: rospy.Publisher
    global_pcd_reference_frame: str

    agents: list[habitat_sim.Agent]
    depth_trunc_meters: list[float]
    o3d_camera_intrinsics: list[o3d.camera.PinholeCameraIntrinsic]
    odom_handles: list[OdomHandle]
    post_proc_publish_threads: list[threading.Thread]

    color_image_raw_pubs: list[rospy.Publisher]
    depth_image_raw_pubs: list[rospy.Publisher]
    semantic_image_raw_pubs: list[rospy.Publisher]
    local_point_cloud_pubs: list[rospy.Publisher]

    def global_point_cloud_pub_cb(self, _):
        msg = PointCloud2()

        msg.header = std_msgs.msg.Header()
        msg.header.frame_id = self.global_pcd_reference_frame
        msg.header.stamp = rospy.Time.now()

        msg.height = 1
        msg.width = len(self.global_point_cloud_points)

        msg.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True

        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width

        msg.data = np.asarray(self.global_point_cloud_points, np.float32).tobytes()

        self.global_point_cloud_pub.publish(msg)

    # TODO: test it
    def get_pcd_transform(
        self, sensor_pose_habitat: habitat_sim.agent.SixDOFPose
    ) -> np.ndarray:
        rotation_world2cam_habitat = sensor_pose_habitat.rotation
        translation_world2cam_habitat = sensor_pose_habitat.position

        rotation_worldframe_mesh2habitat = Rotation.from_matrix(
            [1, 0, 0], [0, 0, -1], [0, 1, 0]
        )
        rotation_camframe_habitat2mesh = Rotation.from_matrix(
            [1, 0, 0], [0, -1, 0], [0, 0, -1]
        )

        # need to calculate
        # world2cam_mesh: world -> cam transform under mesh/o3d convention
        rotation_world2cam_mesh: Rotation = (
            rotation_worldframe_mesh2habitat
            * rotation_world2cam_habitat
            * rotation_camframe_habitat2mesh
        )

        translation_world2cam_mesh: np.ndarray = (
            rotation_worldframe_mesh2habitat * translation_world2cam_habitat
        )

        pcd_transform = np.zeros([4, 4])
        pcd_transform[-1, -1] = 1
        pcd_transform[:3, :3] = rotation_world2cam_mesh.as_matrix()
        pcd_transform[:3, -1] = translation_world2cam_mesh

        return pcd_transform

    def render_loop(self):
        rate = rospy.Rate(self.fps)

        while not rospy.is_shutdown():
            # set agent states according to odom poses
            # and get updated sensor states TODO
            sensor_states = []
            for i in range(self.agent_count):
                agent_state = habitat_sim.AgentState()
                agent_state.position, agent_state.rotation = self.odom_handles[
                    i
                ].get_pose_habitat_frame()
                self.agents[i].set_state(agent_state)

            # let habitat simulator render RGBD and semantic images
            observations = self.simulator.get_sensor_observations(
                range(self.agent_count)
            )

            # parallel post processing and msg publishing
            self.post_proc_publish_threads.clear()
            for i in range(self.agent_count):
                thread = threading.Thread(
                    target=self.post_proc_publish, args=([i, observations[i]])
                )
                thread.start()
                self.post_proc_publish_threads.append(thread)

            for thread in self.post_proc_publish_threads:
                thread.join()

            rate.sleep()

    def post_proc_publish(self, agent_id: int, agent_obs: ObservationDict):
        # note down the current time and create the common header
        timestamp = rospy.Time.now()
        common_header = std_msgs.msg.Header()
        common_header.stamp = timestamp

        # process color image
        # can also use `rgba.astype(np.uint8), encoding="rgba8"` at the cost of higher bandwidth
        rgba = agent_obs["color_sensor" + "_agent_" + str(agent_id)]
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        rgb_msg = CvBridge().cv2_to_imgmsg(rgb.astype(np.uint8), encoding="rgb8")
        rgb_msg.header = common_header

        # process depth image
        # habitat renders depth in meters, the depth array is first truncated
        # rounded to the nearest int, then scaled into millimeters (*1000)
        d = agent_obs["depth_sensor" + "_agent_" + str(agent_id)]
        d = np.rint(d.clip(0.0, self.depth_trunc_meters[agent_id]) * 1000)
        d_msg = CvBridge().cv2_to_imgmsg(d.astype(np.uint16), encoding="16UC1")
        d_msg.header = common_header

        # TODO: semantics

        # process point cloud with rgb, need to transform it into the world frame
        o3d_color = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(d)
        o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d_color,
            depth=o3d_depth,
            depth_scale=1,
            depth_trunc=self.depth_trunc_meters[agent_id],
            convert_rgb_to_intensity=False,
        )
        o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=o3d_rgbd, intrinsic=self.o3d_camera_intrinsics[agent_id]
        )

        # publish messages
        self.color_image_raw_pubs[agent_id].publish(rgb_msg)
        self.depth_image_raw_pubs[agent_id].publish(d_msg)

    def __init__(self) -> None:
        # spawn simulator
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.allow_sliding = rospy.get_param("~allow_sliding")
        sim_cfg.create_renderer = rospy.get_param("~create_renderer")
        sim_cfg.default_agent_id = rospy.get_param("~default_agent_id")
        sim_cfg.enable_gfx_replay_save = rospy.get_param("~enable_gfx_replay_save")
        sim_cfg.enable_physics = rospy.get_param("~enable_physics")
        sim_cfg.force_separate_semantic_scene_graph = rospy.get_param(
            "~force_separate_semantic_scene_graph"
        )
        sim_cfg.gpu_device_id = rospy.get_param("~gpu_device_id")
        sim_cfg.leave_context_with_background_renderer = rospy.get_param(
            "~leave_context_with_background_renderer"
        )
        sim_cfg.load_semantic_mesh = rospy.get_param("~load_semantic_mesh")
        sim_cfg.override_scene_light_defaults = rospy.get_param(
            "~override_scene_light_defaults"
        )
        sim_cfg.scene_light_setup = rospy.get_param("~scene_light_setup")
        sim_cfg.physics_config_file = rospy.get_param("~physics_config_file")
        sim_cfg.random_seed = rospy.get_param("~random_seed")
        sim_cfg.requires_textures = rospy.get_param("~requires_textures")
        sim_cfg.scene_dataset_config_file = rospy.get_param(
            "~scene_dataset_config_file"
        )
        sim_cfg.scene_id = rospy.get_param("~scene_id")
        sim_cfg.use_semantic_textures = rospy.get_param("~use_semantic_textures")

        self.agent_count = rospy.get_param("~agent_count")
        agent_cfgs = []
        for i in range(self.agent_count):
            agent_cfg = habitat_sim.agent.AgentConfiguration()

            agent_cfg.body_type = rospy.get_param("~body_type" + "_agent_" + str(i))
            agent_cfg.height = rospy.get_param("~height" + "_agent_" + str(i))
            agent_cfg.radius = rospy.get_param("~radius" + "_agent_" + str(i))
            agent_cfg.action_space["move_forward"].actuation.amount = rospy.get_param(
                "~forward_step_meter" + "_agent_" + str(i)
            )
            agent_cfg.action_space["turn_left"].actuation.amount = rospy.get_param(
                "~left_step_degree" + "_agent_" + str(i)
            )
            agent_cfg.action_space["turn_right"].actuation.amount = rospy.get_param(
                "~right_step_degree" + "_agent_" + str(i)
            )

            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor" + "_agent_" + str(i)
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            color_sensor_spec.resolution = [
                rospy.get_param("~image_height" + "_agent_" + str(i)),
                rospy.get_param("~image_width" + "_agent_" + str(i)),
            ]
            color_sensor_spec.hfov = rospy.get_param("~hfov" + "_agent_" + str(i))
            color_sensor_spec.position = [
                -rospy.get_param("~sensor_translation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_translation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_translation_x" + "_agent_" + str(i)),
            ]
            color_sensor_spec.orientation = [
                -rospy.get_param("~sensor_rotation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_rotation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_rotation_x" + "_agent_" + str(i)),
            ]

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor" + "_agent_" + str(i)
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor_spec.resolution = [
                rospy.get_param("~image_height" + "_agent_" + str(i)),
                rospy.get_param("~image_width" + "_agent_" + str(i)),
            ]
            depth_sensor_spec.hfov = rospy.get_param("~hfov" + "_agent_" + str(i))
            depth_sensor_spec.position = [
                -rospy.get_param("~sensor_translation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_translation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_translation_x" + "_agent_" + str(i)),
            ]
            depth_sensor_spec.orientation = [
                -rospy.get_param("~sensor_rotation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_rotation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_rotation_x" + "_agent_" + str(i)),
            ]

            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantics_sensor" + "_agent_" + str(i)
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            semantic_sensor_spec.resolution = [
                rospy.get_param("~image_height" + "_agent_" + str(i)),
                rospy.get_param("~image_width" + "_agent_" + str(i)),
            ]
            semantic_sensor_spec.hfov = rospy.get_param("~hfov" + "_agent_" + str(i))
            semantic_sensor_spec.position = [
                -rospy.get_param("~sensor_translation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_translation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_translation_x" + "_agent_" + str(i)),
            ]
            semantic_sensor_spec.orientation = [
                -rospy.get_param("~sensor_rotation_y" + "_agent_" + str(i)),
                +rospy.get_param("~sensor_rotation_z" + "_agent_" + str(i)),
                -rospy.get_param("~sensor_rotation_x" + "_agent_" + str(i)),
            ]

            agent_cfg.sensor_specifications = [
                color_sensor_spec,
                depth_sensor_spec,
                semantic_sensor_spec,
            ]
            agent_cfgs.append(agent_cfg)

        self.simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, agent_cfgs)
        )

        # recompute navigation mesh
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = rospy.get_param("~agent_height")
        navmesh_settings.agent_max_climb = rospy.get_param("~agent_max_climb")
        navmesh_settings.agent_max_slope = rospy.get_param("~agent_max_slope")
        navmesh_settings.agent_radius = rospy.get_param("~agent_radius")
        navmesh_settings.cell_height = rospy.get_param("~cell_height")
        navmesh_settings.cell_size = rospy.get_param("~cell_size")
        navmesh_settings.detail_sample_dist = rospy.get_param("~detail_sample_dist")
        navmesh_settings.detail_sample_max_error = rospy.get_param(
            "~detail_sample_max_error"
        )
        navmesh_settings.edge_max_error = rospy.get_param("~edge_max_error")
        navmesh_settings.edge_max_len = rospy.get_param("~edge_max_len")
        navmesh_settings.filter_ledge_spans = rospy.get_param("~filter_ledge_spans")
        navmesh_settings.filter_low_hanging_obstacles = rospy.get_param(
            "~filter_low_hanging_obstacles"
        )
        navmesh_settings.filter_walkable_low_height_spans = rospy.get_param(
            "~filter_walkable_low_height_spans"
        )
        navmesh_settings.region_merge_size = rospy.get_param("~region_merge_size")
        navmesh_settings.region_min_size = rospy.get_param("~region_min_size")
        navmesh_settings.verts_per_poly = rospy.get_param("~verts_per_poly")
        if not self.simulator.recompute_navmesh(
            self.simulator.pathfinder, navmesh_settings
        ):
            rospy.logwarn("Failed to recompute navigation mesh.")

        # init agents, no need to set initial poses as poses come from odom
        self.agents = []
        for i in range(self.agent_count):
            self.agents.append(self.simulator.initialize_agent(i))

        # o3d read mesh and sample points from it as the global point cloud
        scene_mesh = o3d.io.read_triangle_mesh(rospy.get_param("~scene_id"))
        pcd = scene_mesh.sample_points_uniformly(
            int(rospy.get_param("~mesh_sampling_point"))
        )
        pcd_down = pcd.voxel_down_sample(
            voxel_size=rospy.get_param("~global_pcd_downsample_voxel_size")
        ).translate((0, 0, rospy.get_param("~offset_z")))
        self.global_point_cloud_points = np.asarray(pcd_down.points)

        # also need to get depth truncation settings and camera matrix
        # for post processing
        self.depth_trunc_meters = []
        self.o3d_camera_intrinsics = []
        for i in range(self.agent_count):
            self.depth_trunc_meters.append(
                rospy.get_param("~depth_trunc_meters" + "_agent_" + str(i))
            )

            hfov_deg = rospy.get_param("~hfov" + "_agent_" + str(i))
            hfov = math.radians(hfov_deg)
            w = int(rospy.get_param("~image_width" + "_agent_" + str(i)))
            h = int(rospy.get_param("~image_height" + "_agent_" + str(i)))
            fx = w / (2 * math.tan(hfov / 2))
            fy = fx
            cx = w / 2
            cy = h / 2
            self.o3d_camera_intrinsics.append(
                o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            )

        # register ROS I/O & callbacks below
        self.global_pcd_reference_frame = rospy.get_param("~global_pcd_reference_frame")
        self.global_point_cloud_pub = rospy.Publisher(
            rospy.get_param("~global_pcd_publish_topic"), PointCloud2, queue_size=1
        )
        rospy.Timer(rospy.Duration(1.0), self.global_point_cloud_pub_cb)

        self.odom_handles = []
        self.color_image_raw_pubs = []
        self.depth_image_raw_pubs = []
        self.semantic_image_raw_pubs = []
        self.local_point_cloud_pubs = []
        for i in range(self.agent_count):
            self.odom_handles.append(OdomHandle(i))
            self.color_image_raw_pubs.append(
                rospy.Publisher(
                    rospy.get_param("~color_image_raw_topic" + "_agent_" + str(i)),
                    Image,
                    queue_size=1,
                )
            )
            self.depth_image_raw_pubs.append(
                rospy.Publisher(
                    rospy.get_param("~depth_image_raw_topic" + "_agent_" + str(i)),
                    Image,
                    queue_size=1,
                )
            )
            self.semantic_image_raw_pubs.append(
                rospy.Publisher(
                    rospy.get_param("~semantic_image_raw_topic" + "_agent_" + str(i)),
                    Image,
                    queue_size=1,
                )
            )

        self.fps = rospy.get_param("~render_fps")
        self.post_proc_publish_threads = []


if __name__ == "__main__":
    rospy.init_node("habitat_simulator")
    sync_simulator = SyncSimulator()
    sync_simulator.render_loop()
    rospy.spin()
