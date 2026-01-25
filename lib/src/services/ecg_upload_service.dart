import 'dart:typed_data';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;

class EcgUploadService {
  /// Public so other services can reuse the same base URL.
  static String backendBaseUrl() {
    if (kIsWeb) {
      // Flutter Web (Chrome) -> same machine -> use localhost
      return 'http://127.0.0.1:8000';
    }

    if (Platform.isAndroid) {
      // Android emulator reaching host machine
      return 'http://10.0.2.2:8000';
    }

    // iOS / desktop etc. on same PC
    return 'http://127.0.0.1:8000';
  }

  static String _inferExtToMime(String name) {
    final ext = p.extension(name).toLowerCase();
    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'jpeg';
      case '.png':
        return 'png';
      case '.webp':
        return 'webp';
      default:
        return 'jpeg';
    }
  }

  /// MAIN FUNCTION: Upload ECG file to Django backend
  static Future<http.StreamedResponse> uploadEcg({
    required Uint8List imageBytes,
    required String fileName,
    required String patientName,
    required String patientId,
  }) async {
    final uri = Uri.parse('${backendBaseUrl()}/api/ecg/upload/');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: fileName,
        contentType: MediaType('image', _inferExtToMime(fileName)),
      ),
    );

    request.fields['source'] = 'mobile';
    request.fields['patient_name'] = patientName;
    request.fields['patient_id'] = patientId;

    return await request.send();
  }

  /// STEP 3: Auto-crop via backend OpenCV
  /// POST /api/ecg/auto_crop/ with multipart file => returns image/jpeg bytes
  static Future<Uint8List> autoCropEcg({
    required Uint8List imageBytes,
    required String fileName,
  }) async {
    final uri = Uri.parse('${backendBaseUrl()}/api/ecg/auto_crop/');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: fileName,
        contentType: MediaType('image', _inferExtToMime(fileName)),
      ),
    );

    final streamed = await request.send();
    final respBytes = await streamed.stream.toBytes();

    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      final msg = String.fromCharCodes(respBytes);
      throw Exception('auto_crop failed (${streamed.statusCode}): $msg');
    }

    return Uint8List.fromList(respBytes);
  }
}