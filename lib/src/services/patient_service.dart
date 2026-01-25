// lib/src/services/patient_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/patient.dart';
import 'ecg_upload_service.dart';

class PatientService {
  /// Login with patient_id, or create new if not exists.
  /// Backend rule:
  /// - If ID exists and name does not match => 409 with {"error": "..."}
  static Future<Patient> loginOrCreate({
    required String patientId,
    required String name,
  }) async {
    final uri = Uri.parse(
      '${EcgUploadService.backendBaseUrl()}/api/patient/login_or_create/',
    );

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'patient_id': patientId,
        'name': name,
      }),
    );

    if (response.statusCode == 200 || response.statusCode == 201) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return Patient.fromJson(data);
    }

    String message = 'Login failed.';
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic>) {
        message = (decoded['error'] ?? decoded['message'] ?? message).toString();
      } else {
        message = response.body.toString();
      }
    } catch (_) {
      message = response.body.isNotEmpty ? response.body : message;
    }

    if (response.statusCode == 409) {
      throw Exception(message);
    }

    throw Exception('Server error (${response.statusCode}): $message');
  }

  /// ✅ NEW: Fetch all patients for dashboard browsing
  static Future<List<Map<String, dynamic>>> fetchPatientsRaw() async {
    final uri = Uri.parse('${EcgUploadService.backendBaseUrl()}/api/patients/');
    final response = await http.get(uri);

    if (response.statusCode != 200) {
      throw Exception('Failed to fetch patients (${response.statusCode})');
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final list = (decoded['patients'] as List<dynamic>? ?? [])
        .map((e) => e as Map<String, dynamic>)
        .toList();

    return list;
  }

  /// ✅ NEW: Create patient (ID unique, gender optional)
  static Future<Map<String, dynamic>> createPatient({
    required String patientId,
    required String fullName,
    String? gender, // "M" or "F" or null
  }) async {
    final uri = Uri.parse('${EcgUploadService.backendBaseUrl()}/api/patients/');

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'patient_id': patientId,
        'full_name': fullName,
        'gender': gender, // may be null
      }),
    );

    if (response.statusCode == 201) {
      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      return decoded['patient'] as Map<String, dynamic>;
    }

    String message = 'Create patient failed.';
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic>) {
        message = (decoded['error'] ?? decoded['message'] ?? message).toString();
      } else {
        message = response.body.toString();
      }
    } catch (_) {
      message = response.body.isNotEmpty ? response.body : message;
    }

    if (response.statusCode == 409) {
      throw Exception(message); // patient_id already exists
    }

    throw Exception('Server error (${response.statusCode}): $message');
  }
}