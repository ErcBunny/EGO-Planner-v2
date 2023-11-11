import rospy
import std_msgs
from sensor_msgs.msg import PointCloud2, PointField

import numpy as np
import open3d as o3d
import habitat_sim


class SyncSimulator:
    global_point_cloud_points: np.ndarray
    global_point_cloud_pub: rospy.Publisher

    def global_point_cloud_pub_cb(self, _):
        msg = PointCloud2()

        msg.header = std_msgs.msg.Header()
        msg.header.frame_id = "map"
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

    def __init__(self) -> None:
        rospy.init_node("habitat_simulator")

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

        agent_count = rospy.get_param("~agent_count")
        agent_cfgs = []
        for i in range(agent_count):
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

        simulator = habitat_sim.Simulator(
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
        if not simulator.recompute_navmesh(simulator.pathfinder, navmesh_settings):
            rospy.logwarn("Failed to recompute navigation mesh.")

        # place the agents at the set initial poses
        # TODO

        # o3d read mesh and sample points from it
        scene_mesh = o3d.io.read_triangle_mesh(rospy.get_param("~scene_id"))
        pcd = scene_mesh.sample_points_uniformly(
            int(rospy.get_param("~mesh_sampling_point"))
        )
        pcd_down = pcd.voxel_down_sample(
            voxel_size=rospy.get_param("~global_pcd_downsample_voxel_size")
        )
        self.global_point_cloud_points = np.asarray(pcd_down.points)

        # register ROS I/O & callbacks below
        self.global_point_cloud_pub = rospy.Publisher(
            "~global_cloud", PointCloud2, queue_size=1
        )
        rospy.Timer(rospy.Duration(1.0), self.global_point_cloud_pub_cb)

        # finish initialization
        rospy.spin()


if __name__ == "__main__":
    sync_simulator = SyncSimulator()
