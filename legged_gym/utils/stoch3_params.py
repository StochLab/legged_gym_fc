BODY_LENGTH = 0.541  # hip-to-hip length of the body (in metres)
BODY_WIDTH = 0.203  # hip-to-hip width of the body (in metres)
BODY_HEIGHT = 0.1    # body height (in metres)

ABD_LEN = 0.123  # length of abduction link (metres)
THIGH_LEN = 0.297  # length of thigh link (metres)
SHANK_LEN = 0.347  # length of shank link (metres)

BASE_MASS = 9.7565
ABD_MASS = 0.9451
THIGH_MASS = 2.4957
SHANK_MASS = 0.3792

LEG_MASS = ABD_MASS + THIGH_MASS + SHANK_MASS
ROBOT_MASS = LEG_MASS * 4 + BASE_MASS

ROBOT_IXX = 0.12  # ROBOT_MASS*BODY_WIDTH*BODY_HEIGHT/12
ROBOT_IYY = 0.88  # ROBOT_MASS*BODY_LENGTH*BODY_HEIGHT/12
ROBOT_IZZ = 1.0  # ROBOT_MASS*BODY_LENGTH*BODY_WIDTH/12

NUM_LEGS_ = 4
NUM_MOTORS_PER_LEG_ = 3