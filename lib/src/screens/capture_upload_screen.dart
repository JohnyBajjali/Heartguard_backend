import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_mlkit_document_scanner/google_mlkit_document_scanner.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class CaptureUploadScreen extends StatefulWidget {
  const CaptureUploadScreen({super.key});
  @override
  State<CaptureUploadScreen> createState() => _CaptureUploadScreenState();
}

class _CaptureUploadScreenState extends State<CaptureUploadScreen> {
  Uint8List? _imageBytes;
  String? _fileName;
  bool _isUploading = false;

  // Recent thumbnails (prefilled with placeholders so it isn’t empty)
  final List<ImageProvider> _recent = [];

  @override
  void initState() {
    super.initState();
    _seedPlaceholders();
  }

  Future<void> _seedPlaceholders() async {
    final bytes = await _generateSampleEcgPngBytes();
    setState(() {
      _recent
        ..clear()
        ..addAll(List.generate(4, (_) => MemoryImage(bytes)));
    });
  }

  /* -------------------- IMAGE SELECTION -------------------- */

  Future<void> _scanWithCamera() async {
    final documentOptions = DocumentScannerOptions(
      documentFormat: DocumentFormat.jpeg,
      mode: ScannerMode.filter,
      pageLimit: 1,
      isGalleryImport: false,
    );

    final scanner = DocumentScanner(options: documentOptions);
    try {
      final result = await scanner.scanDocument();
      if (!mounted || result == null) return;

      if (result.images.isNotEmpty) {
        final firstPath = result.images.first;
        final bytes = await File(firstPath).readAsBytes();
        setState(() {
          _imageBytes = bytes;
          _fileName = p.basename(firstPath);
          _rememberRecent(MemoryImage(bytes));
        });
      } else if (result.pdf != null) {
        _showSnack('PDF returned; configure images if you want PNG/JPEG.');
      }
    } catch (e) {
      _showSnack('Scan cancelled or failed: $e');
    } finally {
      await scanner.close();
    }
  }

  Future<void> _importFromGallery() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(picked.path);
      _rememberRecent(MemoryImage(bytes));
    });
  }

  /* -------------------- CROPPING -------------------- */

  Future<void> _manualCrop() async {
    if (_imageBytes == null) return;
    final temp =
    await _bytesToTempFile(_imageBytes!, fileName: _fileName ?? 'ecg.png');

    final cropped = await ImageCropper().cropImage(
      sourcePath: temp.path,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop ECG Area',
          hideBottomControls: false,
          lockAspectRatio: false,
          showCropGrid: true,
        ),
      ],
    );

    if (cropped == null) return;
    final bytes = await cropped.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(cropped.path);
      _rememberRecent(MemoryImage(bytes));
    });
  }

  Future<void> _autoCropEcgRoi() async {
    if (_imageBytes == null) return;
    _showSnack("Auto-crop coming soon — use manual crop for now.");
  }

  /* -------------------- UPLOAD -------------------- */

  Future<void> _uploadToServer() async {
    if (_imageBytes == null) return;
    setState(() => _isUploading = true);

    try {
      final uri = Uri.parse('http://10.0.2.2:8000/api/ecg/upload/');
      final request = http.MultipartRequest('POST', uri);

      final fileName = _fileName ?? 'ecg.jpg';
      request.files.add(http.MultipartFile.fromBytes(
        'file',
        _imageBytes!,
        filename: fileName,
        contentType: MediaType('image', _inferExtToMime(fileName)),
      ));
      request.fields['source'] = 'mobile';

      final resp = await request.send();
      final body = await resp.stream.bytesToString();

      if (resp.statusCode >= 200 && resp.statusCode < 300) {
        _showSnack('✅ Uploaded! Server response: $body');
      } else {
        _showSnack('❌ Upload failed [${resp.statusCode}]: $body');
      }
    } catch (e) {
      _showSnack('Error uploading: $e');
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  /* -------------------- UI -------------------- */

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final hasImage = _imageBytes != null;

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FC),
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            pinned: true,
            expandedHeight: 220,
            backgroundColor: const Color(0xFF0E6B6F),
            foregroundColor: Colors.white,
            title: const Text('ECG Intake'),
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  // NOTE: remove const to avoid "methods in const" on older SDKs
                  DecoratedBox(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        colors: [Color(0xFF0E6B6F), Color(0xFF073A40)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                    ),
                  ),
                  _GridHeaderPainter(opacity: .05),
                  _ECGWaveHeaderPainter(color: Colors.white70),
                  Align(
                    alignment: Alignment.bottomLeft,
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 22),
                      child: Text(
                        'Scan or Import an ECG\nWe’ll crop and prepare it for analysis',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: Colors.white,
                          height: 1.25,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Action cards
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Row(
                children: [
                  Expanded(
                    child: _ActionCard(
                      title: 'Scan / Camera',
                      subtitle: 'Take a photo of the paper',
                      icon: Icons.camera_alt_rounded,
                      color: const Color(0xFF1C7C80),
                      onTap: _scanWithCamera,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _ActionCard(
                      title: 'Import',
                      subtitle: 'From gallery or files',
                      icon: Icons.folder_open_rounded,
                      color: const Color(0xFF0E6B6F),
                      onTap: _importFromGallery,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Preview
          if (hasImage)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: _PreviewCard(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Image.memory(_imageBytes!, fit: BoxFit.cover),
                  ),
                  onCrop: _manualCrop,
                  onAutoTrim: _autoCropEcgRoi,
                ),
              ),
            ),

          // Recent ECGs
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
              child: Row(
                children: [
                  Text('Recent ECGs',
                      style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                  const Spacer(),
                  TextButton(onPressed: () {}, child: const Text('View all')),
                ],
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: SizedBox(
              height: 120,
              child: ListView.separated(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                scrollDirection: Axis.horizontal,
                itemCount: _recent.length,
                separatorBuilder: (_, __) => const SizedBox(width: 12),
                itemBuilder: (_, i) {
                  return _RecentThumb(
                    image: _recent[i],
                    onTap: () {
                      if (_recent[i] is MemoryImage) {
                        final mem = _recent[i] as MemoryImage;
                        setState(() => _imageBytes = mem.bytes);
                      }
                    },
                  );
                },
              ),
            ),
          ),

          // Pipeline chips
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 100),
              child: Wrap(
                spacing: 8,
                runSpacing: -6,
                children: const [
                  _StepChip(label: 'Auto-crop grid'),
                  _StepChip(label: 'Deskew & de-shadow'),
                  _StepChip(label: 'Digitize waveform'),
                  _StepChip(label: 'Send to AI model'),
                ],
              ),
            ),
          ),
        ],
      ),

      // Sticky bottom bar
      bottomNavigationBar: SafeArea(
        minimum: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: ElevatedButton.icon(
          icon: _isUploading
              ? const SizedBox(
              width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.cloud_upload_rounded),
          label: const Padding(
            padding: EdgeInsets.symmetric(vertical: 14),
            child: Text('Upload to Server', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          ),
          onPressed: (!_isUploading && hasImage) ? _uploadToServer : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF0E6B6F),
            disabledBackgroundColor: const Color(0xFF9FB9BC),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          ),
        ),
      ),
    );
  }

  /* -------------------- HELPERS -------------------- */

  void _rememberRecent(ImageProvider provider) {
    _recent.insert(0, provider);
    while (_recent.length > 12) _recent.removeLast();
  }

  Future<File> _bytesToTempFile(Uint8List bytes, {required String fileName}) async {
    final dir = await getTemporaryDirectory();
    final file = File(p.join(dir.path, 'ecg_${DateTime.now().millisecondsSinceEpoch}_$fileName'));
    await file.writeAsBytes(bytes, flush: true);
    return file;
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  String _inferExtToMime(String name) {
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

  /// Generate a PNG bytes placeholder (pink ECG grid + green wave)
  Future<Uint8List> _generateSampleEcgPngBytes() async {
    const w = 800, h = 480;

    final recorder = ui.PictureRecorder();
    // ❌ was: const Rect.fromLTWH(... w.toDouble() ...)
    // ✅ remove const:
    final canvas = ui.Canvas(recorder, Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()));

    const bg = Color(0xFFFFF0F4);
    const dark = Color(0xFFE8A9B8);
    const light = Color(0xFFF8D6DE);

    // ❌ was: final stroke = Paint().style = PaintingStyle.stroke;
    // ✅ use cascade so 'stroke' is a Paint:
    final stroke = Paint()..style = PaintingStyle.stroke;

    canvas.drawPaint(Paint()..color = bg);

    for (double x = 0; x <= w; x += 8) {
      stroke
        ..color = (x % 40 == 0) ? dark : light
        ..strokeWidth = (x % 40 == 0) ? 1.2 : .6;
      canvas.drawLine(Offset(x, 0), Offset(x, h.toDouble()), stroke);
    }
    for (double y = 0; y <= h; y += 8) {
      stroke
        ..color = (y % 40 == 0) ? dark : light
        ..strokeWidth = (y % 40 == 0) ? 1.2 : .6;
      canvas.drawLine(Offset(0, y), Offset(w.toDouble(), y), stroke);
    }

    final wave = Path();
    const mid = h / 2.0;
    wave.moveTo(0, mid);
    for (double x = 0; x < w; x += 160) {
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
    final image = await picture.toImage(w, h);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }
}

/* =================== UI PARTS =================== */

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Ink(
        height: 140,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: Colors.white,
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(.06), blurRadius: 18, offset: const Offset(0, 8)),
          ],
          border: Border.all(color: color.withOpacity(.18), width: 1),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Stack(
            children: [
              Align(alignment: Alignment.topRight, child: Icon(icon, size: 28, color: color)),
              Align(
                alignment: Alignment.bottomLeft,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 4),
                    Text(subtitle, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.black54)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PreviewCard extends StatelessWidget {
  const _PreviewCard({required this.child, required this.onCrop, required this.onAutoTrim});
  final Widget child;
  final VoidCallback onCrop;
  final VoidCallback onAutoTrim;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: Container(
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: theme.colorScheme.primary.withOpacity(.18)),
        ),
        child: Column(
          children: [
            AspectRatio(aspectRatio: 5 / 3, child: child),
            Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      icon: const Icon(Icons.crop),
                      label: const Text('Crop ECG Region'),
                      onPressed: onCrop,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      icon: const Icon(Icons.auto_fix_high),
                      label: const Text('Auto-Trim (beta)'),
                      onPressed: onAutoTrim,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RecentThumb extends StatelessWidget {
  const _RecentThumb({required this.image, required this.onTap});
  final ImageProvider image;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 160,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.black12),
          image: DecorationImage(image: image, fit: BoxFit.cover),
        ),
      ),
    );
  }
}

class _StepChip extends StatelessWidget {
  const _StepChip({required this.label});
  final String label;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label),
      backgroundColor: const Color(0xFFE6F5F5),
      side: const BorderSide(color: Color(0xFFBEE3E3)),
      padding: const EdgeInsets.symmetric(horizontal: 8),
      visualDensity: VisualDensity.compact,
    );
  }
}

/* --------- Header painters (no assets needed) --------- */

class _GridHeaderPainter extends StatelessWidget {
  const _GridHeaderPainter({this.opacity = .07, super.key});
  final double opacity;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _GridPainterImpl(opacity), size: Size.infinite);
  }
}

class _GridPainterImpl extends CustomPainter {
  _GridPainterImpl(this.opacity);
  final double opacity;

  @override
  void paint(Canvas canvas, Size size) {
    final big = Paint()
      ..color = Colors.white.withOpacity(opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final small = Paint()
      ..color = Colors.white.withOpacity(opacity * .4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = .5;

    const bigStep = 40.0;
    const smallStep = 8.0;

    for (double x = 0; x <= size.width; x += smallStep) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), x % bigStep == 0 ? big : small);
    }
    for (double y = 0; y <= size.height; y += smallStep) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), y % bigStep == 0 ? big : small);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _ECGWaveHeaderPainter extends StatelessWidget {
  const _ECGWaveHeaderPainter({required this.color, super.key});
  final Color color;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _ECGWavePainterImpl(color), size: Size.infinite);
  }
}

class _ECGWavePainterImpl extends CustomPainter {
  _ECGWavePainterImpl(this.color);
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final p = Path();
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final mid = size.height * .55;
    final w = size.width;

    p.moveTo(-40, mid);
    for (double x = -40; x < w + 40; x += 160) {
      p.quadraticBezierTo(x + 20, mid - 8, x + 40, mid);
      p.lineTo(x + 55, mid + 4);
      p.lineTo(x + 68, mid - 42);
      p.lineTo(x + 82, mid + 10);
      p.lineTo(x + 95, mid - 6);
      p.quadraticBezierTo(x + 120, mid - 2, x + 140, mid - 4);
      p.quadraticBezierTo(x + 160, mid + 8, x + 180, mid);
    }
    canvas.drawPath(p, paint);
    canvas.drawPath(p, paint..color = color.withOpacity(.25)..strokeWidth = 6);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
