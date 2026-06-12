uint8 IDLE=1
uint8 STARTING=2
uint8 FOLLOWING=3
uint8 STOPPED=4
uint8 FAULT=5

uint8 REASON_NONE=0
uint8 REASON_FOLLOWING_STARTED=1
uint8 REASON_STOPPED=2
uint8 REASON_FAULT=3
uint8 REASON_OBSTACLE_BLOCKED=4

std_msgs/Header header
uint8 phase
uint8 reason
bool blocked
uint32 revision
uint16 code
string message
