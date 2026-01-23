#include "vehicle_control.h"
#include "test_types.h"

/* =========================
 * Main control logic
 * ========================= */
void VehicleControl_Main(
    const VehicleInput_t* input,
    VehicleOutput_t* output
)
{
    uint16_t batteryVoltage_mV;

    /* Defensive programming */
    if ((input == NULL) || (output == NULL))
    {
        return;
    }

    /* Read external dependency (to be stubbed in UT) */
    batteryVoltage_mV = ReadBatteryVoltage_mV();

    /* Default outputs */
    output->engineEnable = false;
    output->coolingFanOn = false;
    output->ecuState     = ECU_STATE_INIT;

    /* =========================
     * Battery check
     * ========================= */
    if (batteryVoltage_mV < 9000U)
    {
        output->ecuState = ECU_STATE_ERROR;
        return; /* Early return */
    }

    /* =========================
     * Temperature handling
     * ========================= */
    if (input->engineTempC >= TEMP_OVERHEAT_C)
    {
        output->ecuState     = ECU_STATE_ERROR;
        output->coolingFanOn = true;
    }
    else if (input->engineTempC >= TEMP_WARNING_C)
    {
        output->ecuState     = ECU_STATE_WARNING;
        output->coolingFanOn = true;
    }
    else
    {
        output->ecuState = ECU_STATE_NORMAL;
    }

    /* =========================
     * Speed validation
     * ========================= */
    if ((input->vehicleSpeedKph > SPEED_MAX_KPH) ||
        (input->vehicleSpeedKph < SPEED_MIN_KPH))
    {
        output->ecuState = ECU_STATE_ERROR;
    }

    /* =========================
     * Engine enable logic
     * MC/DC target condition:
     *  A: accelPedalPressed
     *  B: brakePressed
     *  C: ecuState == NORMAL
     * ========================= */
    if ((input->accelPedalPressed == true) &&
        (input->brakePressed == false) &&
        (output->ecuState == ECU_STATE_NORMAL))
    {
        output->engineEnable = true;
    }
    else
    {
        output->engineEnable = false;
    }

    /* =========================
     * Post-condition safety
     * ========================= */
    switch (output->ecuState)
    {
        case ECU_STATE_ERROR:
            output->engineEnable = false;
            break;

        case ECU_STATE_WARNING:
            /* Engine allowed but cooling must be on */
            output->coolingFanOn = true;
            break;

        case ECU_STATE_NORMAL:
            /* No action */
            break;

        default:
            /* Defensive default */
            output->engineEnable = false;
            output->coolingFanOn = true;
            output->ecuState     = ECU_STATE_ERROR;
            break;
    }
}

void BMS_Monitor(const BatteryInput_t* input, BatteryOutput_t* output) {
    // 1. Check NULL pointers
    if ((input == 0) || (output == 0)) {
        return;
    }

    // 2. Logic IF - ELSE IF - ELSE phức tạp
    if (input->voltage_mV < MIN_VOLTAGE_MV) {
        output->state = BMS_STATE_FAULT;
        output->error_code = ERR_UNDER_VOLTAGE;
    } 
    else if (input->voltage_mV > MAX_VOLTAGE_MV) {
        output->state = BMS_STATE_FAULT;
        output->error_code = ERR_OVER_VOLTAGE;
    } 
    else {
        // Voltage OK, check Temperature (Lồng nhau)
        if (input->temperature_C >= TEMP_CRITICAL) {
            output->state = BMS_STATE_CRITICAL;
            output->fan_speed = 100;
        } 
        else if (input->temperature_C >= TEMP_WARNING) {
            output->state = BMS_STATE_WARNING;
            output->fan_speed = 50;
        } 
        else {
            // Normal case
            output->state = BMS_STATE_NORMAL;
            output->fan_speed = 0;
            output->error_code = ERR_NONE;
        }
    }
}

int SetOperationMode(int request_mode, SystemState_t* sys) {
    int ret_val = 0;
    
    if (sys == 0) return -1;

    switch (request_mode) {
        case MODE_INIT:
            sys->active = 1;
            sys->power_level = 10;
            break;

        case MODE_ECO:
            if (sys->battery_level > 20) {
                sys->power_level = 50;
            } else {
                sys->power_level = 20; // Low power in ECO
            }
            break;

        case MODE_SPORT:
            // Case này phụ thuộc vào biến khác
            if (sys->safety_lock == 0) {
                sys->power_level = 100;
            } else {
                ret_val = -2; // Locked
            }
            break;

        case MODE_STANDBY:
        case MODE_OFF:
            // Fall-through logic (2 case làm chung 1 việc)
            sys->active = 0;
            sys->power_level = 0;
            break;

        default:
            ret_val = -1; // Invalid mode
            break;
    }
    
    return ret_val;
}

void ProcessSensors(int* data, int count, SensorStatus_t* status) {
    int i;
    
    if (data == 0 || status == 0) return;
    
    // Test logic vòng lặp (Tool MC/DC thường bỏ qua loop body nhưng vẫn cần parse được)
    for (i = 0; i < count; i++) {
        
        // Test gọi hàm ngoài -> Cần sinh TEST.VALUE:uut_prototype_stubs...
        int isValid = ValidateSensor(i);
        
        if (isValid) {
            // Test toán tử 3 ngôi
            status->level[i] = (data[i] > 1000) ? LEVEL_HIGH : LEVEL_LOW;
        } else {
            status->level[i] = LEVEL_INVALID;
            LogError(0xFF); // Gọi stub void
        }
    }
    
    // Test logic phức tạp với toán tử logic
    if ((status->level[0] == LEVEL_HIGH) && (status->level[1] == LEVEL_HIGH) || (count > 5)) {
        status->global_alert = 1;
    } else {
        status->global_alert = 0;
    }
}

int CalculateAverage(const int* values, int size) {
    int sum = 0;
    int count = 0;
    int i; 

    if (values == 0) return 0;
    
    // Test: Tool cần tự gán size = 1 để vào vòng lặp
    for (i = 0; i < size; i++) {
        // Test: values[i] > 0 -> Cần sinh values[0] = 1
        if (values[i] > 0) {
            sum += values[i];
            count++;
        }
    }
    
    return (count > 0) ? (sum / count) : 0;
}

int FindCriticalSensor(const SensorData_t* sensors, int num_sensors) {
    int idx = -1;
    int k;

    if (sensors == 0) return -1;

    for (k = 0; k < num_sensors; k++) {
        // Case 1: Tìm thấy lỗi critical -> Break ngay
        if (sensors[k].status == STATUS_CRITICAL) {
            idx = k;
            break; 
        }
        
        // Case 2: Logic lồng nhau (Nested IF)
        if (sensors[k].value > 100) {
             if (sensors[k].type == TYPE_TEMP) {
                 idx = k;
                 // Cần test xem tool có cover được nhánh này không
                 break;
             }
        }
    }
    return idx;
}

void CheckSystemIntegrity(int* modules, int total, SystemState_t* status) {
    int errors = 0;
    int warnings = 0;
    int j;
    
    if (modules == 0 || status == 0) return;

    for (j = 0; j < total; j++) {
        // Logic phức tạp: (A && B) || C
        // C ở đây là (j > 5). Tool phải gán total > 6 để j chạy được tới 6.
        if ((modules[j] > 100 && modules[j] < 200) || (j > 5)) {
            warnings++;
        }
        
        if (modules[j] == 0) {
            errors++;
        }
    }
    
    // Logic tổng hợp sau vòng lặp
    if (errors > 0 || warnings > 3) {
        status->active = 0; // Fail
    } else {
        status->active = 1; // OK
    }
}


/* =========================================================================
 * 1. Test WHILE Loop: MonitorCharging
 * Logic: Giả lập quá trình sạc pin. Vòng lặp chạy khi pin < 100%.
 * Sử dụng: MAX_VOLTAGE_MV
 * ========================================================================= */
void MonitorCharging(SystemState_t* sys, int current_voltage) {
    if (sys == NULL) return;

    // Tool cần nhận diện điều kiện vào vòng lặp: sys->battery_level < 100
    // Smart Loop Value nên gán battery_level = 99 hoặc nhỏ hơn
    while (sys->battery_level < 100) {
        
        // Safety Break: Nếu điện áp quá cao -> Ngắt sạc ngay
        if (current_voltage > MAX_VOLTAGE_MV) {
            sys->active = 0; // Tắt hệ thống
            break;
        }

        // Giả lập tăng dung lượng pin
        sys->battery_level++;
    }
}

/* =========================================================================
 * 2. Test DO-WHILE Loop: ResetSensorLevels
 * Logic: Quét mảng cảm biến, reset các sensor báo HIGH về LOW.
 * Sử dụng: MAX_SENSORS (10), LEVEL_HIGH, LEVEL_LOW
 * ========================================================================= */
int ResetSensorLevels(SensorStatus_t* status) {
    int i = 0;
    int reset_count = 0;

    if (status == NULL) return -1;

    // Do-while luôn chạy ít nhất 1 lần
    do {
        // Nested IF bên trong vòng lặp
        if (status->level[i] == LEVEL_HIGH) {
            status->level[i] = LEVEL_LOW;
            reset_count++;
        }
        i++;
    } while (i < 5);

    return reset_count;
}


void ProcessSensorData(SensorData_u data, int type) {
    switch (type) {
        case DATA_TYPE_RAW:
            // Xử lý như số nguyên
            if (data.raw_value > 4000) {
                printf("[WARN] Raw value too high: %d\n", data.raw_value);
            } else {
                printf("Raw Value: %d\n", data.raw_value);
            }
            break;

        case DATA_TYPE_VOLTAGE:
            // Xử lý như số thực
            printf("Voltage: %.2f V\n", data.voltage);
            break;

        case DATA_TYPE_ERROR:
            // Xử lý như mã lỗi 8-bit
            printf("Sensor Error Code: 0x%02X\n", data.error_code);
            break;

        default:
            printf("Unknown Data Type\n");
            break;
    }
}

int maimailaem() {
    SensorData_u myData;

    // Case 1: Gửi giá trị Raw
    myData.raw_value = 2048;
    ProcessSensorData(myData, DATA_TYPE_RAW);

    // Case 2: Gửi giá trị Voltage (Lưu ý: Ghi đè lên vùng nhớ cũ của raw_value)
    myData.voltage = 3.3; 
    ProcessSensorData(myData, DATA_TYPE_VOLTAGE);

    return 0;
}

int CheckLogic(int a,int b) {
    if(a>5)
    {
        b = randomGenerateNumber();
    }

    if((a > 15) && b > 10)
        printf("pass");
    return 0;
}