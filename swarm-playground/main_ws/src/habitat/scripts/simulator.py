import rospy

if __name__ == "__main__":
    rospy.init_node("habitat_simulator")
    
    print(rospy.get_param("~body_type_agent0"))