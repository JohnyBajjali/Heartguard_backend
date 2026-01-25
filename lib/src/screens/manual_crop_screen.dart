import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:crop_your_image/crop_your_image.dart';

class ManualCropScreen extends StatefulWidget {
  final Uint8List imageBytes;
  const ManualCropScreen({super.key, required this.imageBytes});

  @override
  State<ManualCropScreen> createState() => _ManualCropScreenState();
}

class _ManualCropScreenState extends State<ManualCropScreen> {
  final CropController _controller = CropController();
  bool _cropping = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF101214),
      appBar: AppBar(
        backgroundColor: const Color(0xFF101214),
        foregroundColor: Colors.white,
        elevation: 0,
        title: const Text('Manual Crop'),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Stack(
        children: [
          Positioned.fill(
            child: Crop(
              controller: _controller,
              image: widget.imageBytes,
              withCircleUi: false,
              baseColor: const Color(0xFF101214),
              maskColor: Colors.black.withOpacity(0.55),
              cornerDotBuilder: (size, edgeAlignment) => DotControl(
                color: const Color(0xFF1C7C80),
              ),
              onCropped: (croppedBytes) {
                if (!mounted) return;
                setState(() => _cropping = false);
                Navigator.pop(context, croppedBytes);
              },
            ),
          ),
          if (_cropping)
            const Positioned.fill(
              child: ColoredBox(
                color: Color(0x66000000),
                child: Center(child: CircularProgressIndicator()),
              ),
            ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.close),
                  label: const Text('Cancel'),
                  onPressed: _cropping ? null : () => Navigator.pop(context),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: BorderSide(color: Colors.white.withOpacity(0.25)),
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  icon: const Icon(Icons.check),
                  label: const Text('Done'),
                  onPressed: _cropping
                      ? null
                      : () {
                    setState(() => _cropping = true);
                    _controller.crop();
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0E6B6F),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
