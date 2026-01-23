#ifndef TEST_TYPES_H
#define TEST_TYPES_H

#define MAX_TEMP 100
#define MIN_TEMP 0
#define ERROR_CODE 404
#define ADMIN_ID 99
#define CMD_INIT    10
#define CMD_START   20
#define CMD_STOP    30
#define CMD_ERROR   99

/* =========================================================
 * KHẮC PHỤC LỖI THƯ VIỆN <stdint.h> và <stdbool.h>
 * Thay vì include, ta tự định nghĩa (typedef) thủ công.
 * ========================================================= */
#ifndef NULL
    #define NULL ((void*)0)
#endif
/* 1. Định nghĩa các kiểu số nguyên (Integer Types) */
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef signed char        int8_t;
typedef signed short       int16_t;
typedef signed int         int32_t;

/* 2. Định nghĩa kiểu Boolean (nếu chưa có) */
#ifndef __cplusplus
    #ifndef bool
        typedef int bool;
        #define true 1
        #define false 0
    #endif
#endif

/* =========================================================
 * CÁC ĐỊNH NGHĨA CỦA DỰ ÁN (Project Defines)
 * ========================================================= */

#define MIN_VOLTAGE_MV 3000
#define MAX_VOLTAGE_MV 4200
#define TEMP_WARNING 45
#define TEMP_CRITICAL 60
#define TEMP_OVERHEAT_C 120
#define TEMP_WARNING_C 100
#define SPEED_MAX_KPH 200
#define SPEED_MIN_KPH 0

/* Error Codes */
#define ERR_NONE 0
#define ERR_UNDER_VOLTAGE 101
#define ERR_OVER_VOLTAGE 102

/* Operation Modes */
#define MODE_OFF 0
#define MODE_INIT 1
#define MODE_STANDBY 2
#define MODE_ECO 3
#define MODE_SPORT 4

/* Sensor Levels */
#define LEVEL_INVALID 0
#define LEVEL_LOW 1
#define LEVEL_HIGH 2

/* Sensor & System Defines (Cho hàm mới) */
#define STATUS_OK 0
#define STATUS_CRITICAL 99
#define TYPE_TEMP 1
#define TYPE_SPEED 2

/* BMS States */
#define BMS_STATE_NORMAL 0
#define BMS_STATE_WARNING 1
#define BMS_STATE_CRITICAL 2
#define BMS_STATE_FAULT 3

// Định nghĩa các loại dữ liệu để phân biệt (Tagged Union)
#define DATA_TYPE_RAW     0
#define DATA_TYPE_VOLTAGE 1
#define DATA_TYPE_ERROR   2

/* =========================================================
 * CÁC STRUCT
 * ========================================================= */

typedef struct {
    int voltage_mV;
    int temperature_C;
} BatteryInput_t;

typedef struct {
    int state;
    int error_code;
    int fan_speed;
} BatteryOutput_t;

typedef struct {
    uint16_t vehicleSpeedKph;  // Dùng uint16_t đã typedef ở trên
    int engineTempC;
    bool accelPedalPressed;    // Dùng bool đã typedef
    bool brakePressed;
    int voltage_mV;            // Thêm field này nếu BMS_Monitor cần
    int temperature_C;         // Thêm field này nếu BMS_Monitor cần
} VehicleInput_t;

typedef struct {
    bool engineEnable;
    bool coolingFanOn;
    int ecuState;
    int state;                 // Cho BMS output
    int error_code;
    int fan_speed;
} VehicleOutput_t;

typedef struct {
    int id;
    int status;
    int value;
    int type;
} SensorData_t;

typedef struct {
    int active;
    int power_level;
    int battery_level;
    int safety_lock;
} SystemState_t;

typedef struct {
    int level[10];
    int global_alert;
} SensorStatus_t;

/* Enum ECU State (nếu cần) */
enum {
    ECU_STATE_INIT = 0,
    ECU_STATE_NORMAL = 1,
    ECU_STATE_WARNING = 2,
    ECU_STATE_ERROR = 3
};

/* Union ECU State (nếu cần) */
typedef union {
    uint16_t raw_value;   // Giá trị thô (VD: 0 - 4095)
    float voltage;        // Giá trị quy đổi (VD: 3.3V)
    uint8_t error_code;   // Mã lỗi (nếu có)
} SensorData_u;

#endif // TEST_TYPES_H