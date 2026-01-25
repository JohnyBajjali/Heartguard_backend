import 'dart:typed_data';
import 'package:flutter/material.dart';

class WebManualCropScreen extends StatefulWidget {
  final Uint8List imageBytes;

  const WebManualCropScreen({super.key, required this.imageBytes});

  @override
  State<WebManualCropScreen> createState() => _WebManualCropScreenState();
}

class _WebManualCropScreenState extends State<WebManualCropScreen> {
  final TransformationController _controller = TransformationController();

  // Simple crop box (percentage based on screen)
  Rect _cropRect = const Rect.fromLTWH(80, 120, 800, 400);

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final img = Image.memory(widget.imageBytes);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Manual Crop (Web)'),
        actions: [
          TextButton(
            onPressed: () {
              // For now return the original bytes (we will do real cropping next step)
              // This removes overflow and gives correct full-page UI.
              Navigator.pop(context, widget.imageBytes);
            },
            child: const Text('Done', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: InteractiveViewer(
              transformationController: _controller,
              maxScale: 8,
              minScale: 0.5,
              child: Center(child: img),
            ),
          ),

          // Crop overlay
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(
                painter: _CropOverlayPainter(_cropRect),
              ),
            ),
          ),

          // draggable crop box (basic)
          Positioned(
            left: _cropRect.left,
            top: _cropRect.top,
            width: _cropRect.width,
            height: _cropRect.height,
            child: GestureDetector(
              onPanUpdate: (d) {
                setState(() {
                  _cropRect = _cropRect.shift(d.delta);
                });
              },
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(width: 2, color: Colors.tealAccent),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CropOverlayPainter extends CustomPainter {
  final Rect rect;
  _CropOverlayPainter(this.rect);

  @override
  void paint(Canvas canvas, Size size) {
    final overlayPaint = Paint()..color = Colors.black.withOpacity(0.55);

    // draw outside-rect overlay using even-odd path (works on Web)
    final path = Path()
      ..fillType = PathFillType.evenOdd
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRect(rect);

    canvas.drawPath(path, overlayPaint);
  }

  @override
  bool shouldRepaint(covariant _CropOverlayPainter oldDelegate) => oldDelegate.rect != rect;
}
