import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class EcgPlaceholderService {
  /// Generates a simple ECG-like PNG (grid + wave) as bytes.
  static Future<Uint8List> generateSampleEcgPngBytes({
    int width = 800,
    int height = 480,
  }) async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(
      recorder,
      Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
    );

    const bg = Color(0xFFFFF0F4);
    const dark = Color(0xFFE8A9B8);
    const light = Color(0xFFF8D6DE);

    final stroke = Paint()..style = PaintingStyle.stroke;

    // Background
    canvas.drawPaint(Paint()..color = bg);

    // Grid
    for (double x = 0; x <= width; x += 8) {
      stroke
        ..color = (x % 40 == 0) ? dark : light
        ..strokeWidth = (x % 40 == 0) ? 1.2 : .6;
      canvas.drawLine(Offset(x, 0), Offset(x, height.toDouble()), stroke);
    }
    for (double y = 0; y <= height; y += 8) {
      stroke
        ..color = (y % 40 == 0) ? dark : light
        ..strokeWidth = (y % 40 == 0) ? 1.2 : .6;
      canvas.drawLine(Offset(0, y), Offset(width.toDouble(), y), stroke);
    }

    // Simple ECG wave
    final wave = Path();
    final mid = height / 2.0;
    wave.moveTo(0, mid);
    for (double x = 0; x < width; x += 160) {
      wave.quadraticBezierTo(x + 20, mid - 12, x + 40, mid);
      wave.lineTo(x + 58, mid + 6);
      wave.lineTo(x + 74, mid - 60);
      wave.lineTo(x + 92, mid + 16);
      wave.quadraticBezierTo(x + 130, mid - 4, x + 160, mid);
    }

    canvas.drawPath(
      wave,
      Paint()
        ..color = const Color(0xFF0E6B6F)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );

    final picture = recorder.endRecording();
    final image = await picture.toImage(width, height);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }
}
