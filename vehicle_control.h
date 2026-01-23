#ifndef VEHICLE_CONTROL_H
#define VEHICLE_CONTROL_H

#include "test_types.h"

/* =========================================================================
 * KHAI BÁO PROTOTYPE (Phải khớp chính xác kiểu dữ liệu trong .c)
 * ========================================================================= */

// 1. VehicleControl_Main
// [FIX] Sửa từ void* thành VehicleInput_t* và VehicleOutput_t*
void VehicleControl_Main(const VehicleInput_t* input, VehicleOutput_t* output);

// 2. BMS_Monitor
void BMS_Monitor(const BatteryInput_t* input, BatteryOutput_t* output);

// 3. SetOperationMode
int SetOperationMode(int request_mode, SystemState_t* sys);

// 4. ProcessSensors
void ProcessSensors(int* data, int count, SensorStatus_t* status);

// 5. Các hàm test vòng lặp (Mới)
int CalculateAverage(const int* values, int size);
int FindCriticalSensor(const SensorData_t* sensors, int num_sensors);
void CheckSystemIntegrity(int* modules, int total, SystemState_t* status);

/* =========================================================================
 * STUB FUNCTIONS (Các hàm phụ thuộc)
 * ========================================================================= */
int ReadBatteryVoltage_mV(void);
int ValidateSensor(int index);
void LogError(int code);
void MonitorCharging(SystemState_t* sys, int current_voltage);
int ResetSensorLevels(SensorStatus_t* status);
int randomGenerateNumber();

#endif // VEHICLE_CONTROL_H