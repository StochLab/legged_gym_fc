<?xml version="1.0" encoding="utf-8"?>
<!-- This URDF was automatically created by SolidWorks to URDF Exporter! Originally created by Stephen Brawner (brawner@gmail.com) 
     Commit Version: 1.5.1-0-g916b5db  Build Version: 1.5.7152.31018
     For more information, please see http://wiki.ros.org/sw_urdf_exporter -->
<robot
  name="stoch3_description">
  <link
    name="base_link">
    <inertial>
      <origin
        xyz="-0.10238 -1.8464E-05 0.0026538"
        rpy="0 0 0" />
      <mass
        value="7.3" />
      <inertia
        ixx="0.0090197"
        ixy="2.2679E-06"
        ixz="6.4897E-05"
        iyy="0.0094033"
        iyz="-2.1628E-08"
        izz="0.01573" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/base_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.75294 0.75294 0.75294 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/base_link.STL" />
      </geometry>
    </collision>
  </link>
  <link
    name="fl_abd_link">
    <inertial>
      <origin
        xyz="0.070269 -0.036118 -2.5973E-07"
        rpy="0 0 0" />
      <mass
        value="0.978" />
      <inertia
        ixx="0.00045941"
        ixy="-6.723E-07"
        ixz="3.3247E-08"
        iyy="0.00059328"
        iyz="5.8817E-09"
        izz="0.00046214" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_abd_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_abd_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fl_abd_joint"
    type="revolute">
    <origin
      xyz="0.18598 0.10004 -0.0005"
      rpy="0 0 0.00021134" />
    <parent
      link="base_link" />
    <child
      link="fl_abd_link" />
    <axis
      xyz="1 0 0" />
    <limit
      lower="-1.5708"
      upper="1.5708"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="fl_thigh_link">
    <inertial>
      <origin
        xyz="1.3632E-05 0.076425 -0.072527"
        rpy="0 -1.57078 0" />
      <mass
        value="1.547" />
      <inertia
        ixx="0.00031538"
        ixy="-4.0486E-06"
        ixz="-4.8939E-07"
        iyy="0.0031011"
        iyz="6.4773E-11"
        izz="0.0028703" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_thigh_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_thigh_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fl_hip_joint"
    type="revolute">
    <origin
      xyz="0.074 -0.01975 0"
      rpy="0 0 0" />
    <parent
      link="fl_abd_link" />
    <child
      link="fl_thigh_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="1.57078"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="fl_shank_link">
    <inertial>
      <origin
        xyz="-0.01785 2.469E-07 -0.18227 "
        rpy="0 1.57078 0" />
      <mass
        value="0.2" />
      <inertia
        ixx="2.1525E-05"
        ixy="-8.02E-10"
        ixz="2.6374E-05"
        iyy="0.00042744"
        iyz="-2.0344E-10"
        izz="0.00040935" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_shank_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fl_shank_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fl_knee_joint"
    type="revolute">
    <origin
      xyz="0 0.1385 -0.297"
      rpy="0 0 0" />
    <parent
      link="fl_thigh_link" />
    <child
      link="fl_shank_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="0"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="fr_abd_link">
    <inertial>
      <origin
        xyz="0.07027 0.036118 -2.9553E-07"
        rpy="0 0 0" />
      <mass
        value="0.978" />
      <inertia
        ixx="0.00045941"
        ixy="6.965E-07"
        ixz="-3.3418E-08"
        iyy="0.00059328"
        iyz="-5.8767E-09"
        izz="0.00046214" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_abd_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_abd_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fr_abd_joint"
    type="revolute">
    <origin
      xyz="0.18602 -0.099961 -0.0005"
      rpy="0 0 0.00021134" />
    <parent
      link="base_link" />
    <child
      link="fr_abd_link" />
    <axis
      xyz="1 0 0" />
    <limit
      lower="-1.5708"
      upper="1.5708"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="fr_thigh_link">
    <inertial>
      <origin
        xyz="-1.3633E-05 -0.076425 -0.072527"
        rpy="0 -1.57078 0" />
      <mass
        value="1.547" />
      <inertia
        ixx="0.00031538"
        ixy="4.0486E-06"
        ixz="4.8938E-07"
        iyy="0.0031011"
        iyz="6.4797E-11"
        izz="0.0028703" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_thigh_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_thigh_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fr_hip_joint"
    type="revolute">
    <origin
      xyz="0.074 0.01975 0"
      rpy="0 0 0" />
    <parent
      link="fr_abd_link" />
    <child
      link="fr_thigh_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="1.57078"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="fr_shank_link">
    <inertial>
      <origin
        xyz="-0.017857 2.469E-07 -0.18227"
        rpy="0 1.57078 0" />
      <mass
        value="0.2" />
      <inertia
        ixx="2.1525E-05"
        ixy="-8.02E-10"
        ixz="2.6374E-05"
        iyy="0.00042744"
        iyz="-2.0344E-10"
        izz="0.00040935" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_shank_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/fr_shank_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="fr_knee_joint"
    type="revolute">
    <origin
      xyz="0 -0.1385 -0.297"
      rpy="0 0 0" />
    <parent
      link="fr_thigh_link" />
    <child
      link="fr_shank_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="0"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="bl_abd_link">
    <inertial>
      <origin
        xyz="-0.06995 -0.037322 -1.8681E-06"
        rpy="0 0 0" />
      <mass
        value="0.978" />
      <inertia
        ixx="0.00039025"
        ixy="5.9364E-08"
        ixz="1.1626E-07"
        iyy="0.00050148"
        iyz="5.3635E-09"
        izz="0.00037524" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_abd_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_abd_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="bl_abd_joint"
    type="revolute">
    <origin
      xyz="-0.18602 0.099961 -0.0005"
      rpy="0 0 0.00021134" />
    <parent
      link="base_link" />
    <child
      link="bl_abd_link" />
    <axis
      xyz="1 0 0" />
    <limit
      lower="-1.5708"
      upper="1.5708"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="bl_thigh_link">
    <inertial>
      <origin
        xyz="1.3632E-05 0.076425 -0.072527"
        rpy="0 -1.57078 0" />
      <mass
        value="1.547" />
      <inertia
        ixx="0.00031538"
        ixy="-4.0486E-06"
        ixz="-4.8939E-07"
        iyy="0.0031011"
        iyz="6.4774E-11"
        izz="0.0028703" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_thigh_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_thigh_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="bl_hip_joint"
    type="revolute">
    <origin
      xyz="-0.074 -0.01975 0"
      rpy="0 0 0" />
    <parent
      link="bl_abd_link" />
    <child
      link="bl_thigh_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="1.57078"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="bl_shank_link">
    <inertial>
      <origin
        xyz="-0.017857 2.469E-07 -0.18227"
        rpy="0 1.57078 0" />
      <mass
        value="0.2" />
      <inertia
        ixx="2.1525E-05"
        ixy="-8.02E-10"
        ixz="2.6374E-05"
        iyy="0.00042744"
        iyz="-2.0344E-10"
        izz="0.00040935" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_shank_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/bl_shank_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="bl_knee_joint"
    type="revolute">
    <origin
      xyz="0 0.1385 -0.297"
      rpy="0 0 0" />
    <parent
      link="bl_thigh_link" />
    <child
      link="bl_shank_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="0"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="br_abd_link">
    <inertial>
      <origin
        xyz="-0.070173 0.037288 -1.5868E-07"
        rpy="0 0 0" />
      <mass
        value="0.978" />
      <inertia
        ixx="0.00038996"
        ixy="-9.0848E-07"
        ixz="-8.0338E-08"
        iyy="0.00050185"
        iyz="-1.0951E-09"
        izz="0.00037578" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_abd_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_abd_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="br_abd_joint"
    type="revolute">
    <origin
      xyz="-0.18598 -0.10004 -0.0005"
      rpy="0 0 0.00021134" />
    <parent
      link="base_link" />
    <child
      link="br_abd_link" />
    <axis
      xyz="1 0 0" />
    <limit
      lower="-1.5708"
      upper="1.5708"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="br_thigh_link">
    <inertial>
      <origin
        xyz="-1.3633E-05 -0.076425 -0.072527"
        rpy="0 -1.57078 0" />
      <mass
        value="1.547" />
      <inertia
        ixx="0.00031538"
        ixy="4.0486E-06"
        ixz="4.8938E-07"
        iyy="0.0031011"
        iyz="6.4788E-11"
        izz="0.0028703" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_thigh_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 -1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_thigh_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="br_hip_joint"
    type="revolute">
    <origin
      xyz="-0.074 0.01975 0"
      rpy="0 0 0" />
    <parent
      link="br_abd_link" />
    <child
      link="br_thigh_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="1.57078"
      effort="100"
      velocity="100" />
  </joint>
  <link
    name="br_shank_link">
    <inertial>
      <origin
        xyz="-0.017857 2.469E-07 -0.18227"
        rpy="0 1.57078 0" />
      <mass
        value="0.2" />
      <inertia
        ixx="2.1525E-05"
        ixy="-8.02E-10"
        ixz="2.6374E-05"
        iyy="0.00042744"
        iyz="-2.0344E-10"
        izz="0.00040935" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_shank_link.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.79216 0.81961 0.93333 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 1.57078 0" />
      <geometry>
        <mesh
          filename="package://stoch3_description/meshes/br_shank_link.STL" />
      </geometry>
    </collision>
  </link>
  <joint
    name="br_knee_joint"
    type="revolute">
    <origin
      xyz="0 -0.1385 -0.297"
      rpy="0 0 0" />
    <parent
      link="br_thigh_link" />
    <child
      link="br_shank_link" />
    <axis
      xyz="0 1 0" />
    <limit
      lower="-3.14159"
      upper="0"
      effort="100"
      velocity="100" />
  </joint>





  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <legacyModeNS>true</legacyModeNS>
      <robotNamespace>stoch3</robotNamespace>
    </plugin>
  </gazebo>

  <transmission name="fl_abd_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fl_abd_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fl_abd_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>


  <transmission name="fl_hip_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fl_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fl_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="fl_knee_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fl_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fl_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="fr_abd_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fr_abd_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fr_abd_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="fr_hip_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fr_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fr_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="fr_knee_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="fr_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="fr_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="bl_abd_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="bl_abd_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="bl_abd_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="bl_hip_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="bl_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="bl_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="bl_knee_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="bl_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="bl_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="br_abd_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="br_abd_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="br_abd_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>


  <transmission name="br_hip_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="br_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="br_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="br_knee_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="br_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="br_knee_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
