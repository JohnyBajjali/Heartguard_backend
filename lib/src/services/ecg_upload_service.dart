import 'dart:typed_data';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;

class EcgUploadService {
  /// URL logic that was previously inside your screen
  static String _backendBaseUrl() {
    if (kIsWeb) {
      return 'http://192.168.1.214:8000';
    }

    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000';
    }

    return 'http://192.168.1.214:8000';
  }

  /// This was also inside your screen – moved here
  static String _inferExtToMime(String name) {
    final ext = p.extension(name).toLowerCase();

    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'jpeg';
      case '.webp':
        return 'webp';
      case '.bmp':
        return 'bmp';
      case '.heic':
        return 'heic';
      default:
        return 'png';
    }
  }

  /// MAIN FUNCTION: Upload ECG file to Django backend
  static Future<http.StreamedResponse> uploadEcg({
    required Uint8List imageBytes,
    required String fileName,
    required String patientName,
  }) async {
    final uri = Uri.parse('${_backendBaseUrl()}/api/ecg/upload/');
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

    return await request.send();
  }
}
