import 'dart:typed_data';
import 'dart:io' show File, Platform;

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:google_mlkit_document_scanner/google_mlkit_document_scanner.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../screens/manual_crop_screen.dart';
import '../services/ecg_upload_service.dart';
import '../widgets/action_card.dart';
import '../widgets/preview_card.dart';
import '../widgets/recent_thumb.dart';
import '../painters/ecg_header_painters.dart';

class CaptureUploadScreen extends StatefulWidget {
  final String patientId;
  final String patientName;
  final bool startWithScanner;

  const CaptureUploadScreen({
    super.key,
    required this.patientId,
    required this.patientName,
    this.startWithScanner = false,
  });

  @override
  State<CaptureUploadScreen> createState() => _CaptureUploadScreenState();
}

class _CaptureUploadScreenState extends State<CaptureUploadScreen> {
  Uint8List? _imageBytes;
  String? _fileName;
  String? _imagePath;
  bool _isUploading = false;

  final List<ImageProvider> _recent = [];

  @override
  void initState() {
    super.initState();

    if (widget.startWithScanner) {
      Future.microtask(() async {
        if (!mounted) return;
        await _scanWithCamera();
      });
    }
  }

  /* -------------------- IMAGE PICK / SCAN -------------------- */

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source, imageQuality: 95);
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    if (!mounted) return;

    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(picked.path);
      _imagePath = picked.path;
      _rememberRecent(MemoryImage(bytes));
    });
  }

  Future<void> _scanWithCamera() async {
    // WEB: camera
    if (kIsWeb) {
      await _pickImage(ImageSource.camera);
      return;
    }

    final isMobile = Platform.isAndroid || Platform.isIOS;

    // DESKTOP: camera/scanner not reliable → pick file
    if (!isMobile) {
      await _pickImage(ImageSource.gallery);
      return;
    }

    // ANDROID/iOS: MLKit scanner
    final scanner = DocumentScanner(
      options: DocumentScannerOptions(
        documentFormat: DocumentFormat.jpeg,
        mode: ScannerMode.filter,
        pageLimit: 1,
        isGalleryImport: false,
      ),
    );

    try {
      final result = await scanner.scanDocument();
      if (!mounted) return;

      if (result.images.isNotEmpty) {
        final firstPath = result.images.first;
        final bytes = await File(firstPath).readAsBytes();

        if (!mounted) return;
        setState(() {
          _imageBytes = bytes;
          _fileName = p.basename(firstPath);
          _imagePath = firstPath;
          _rememberRecent(MemoryImage(bytes));
        });
      } else if (result.pdf != null) {
        _showSnack('Scanner returned PDF; configure scanner to return images.');
      }
    } catch (e) {
      if (mounted) _showSnack('Scan cancelled or failed: $e');
    } finally {
      await scanner.close();
    }
  }

  Future<void> _importFromGallery() => _pickImage(ImageSource.gallery);

  /* -------------------- CROP -------------------- */

  Future<void> _manualCrop() async {
    if (_imageBytes == null) return;

    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    // ✅ Web + Desktop: real crop screen (resizable + Done)
    if (!isMobile) {
      final croppedBytes = await Navigator.of(context).push<Uint8List>(
        MaterialPageRoute(
          builder: (_) => ManualCropScreen(imageBytes: _imageBytes!),
        ),
      );

      if (croppedBytes == null) return;

      setState(() {
        _imageBytes = croppedBytes;
        _fileName = _fileName ?? 'ecg.jpg';
        _rememberRecent(MemoryImage(croppedBytes));
      });
      return;
    }

    // ✅ Mobile: image_cropper
    final temp = await _bytesToTempFile(
      _imageBytes!,
      fileName: _fileName ?? 'ecg.jpg',
    );

    final cropped = await ImageCropper().cropImage(
      sourcePath: temp.path,
      compressFormat: ImageCompressFormat.jpg,
      compressQuality: 95,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop ECG Area',
          hideBottomControls: false,
          lockAspectRatio: false,
          showCropGrid: true,
        ),
        IOSUiSettings(title: 'Crop ECG Area'),
      ],
    );

    if (cropped == null) return;

    final bytes = await cropped.readAsBytes();
    if (!mounted) return;

    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(cropped.path);
      _imagePath = cropped.path;
      _rememberRecent(MemoryImage(bytes));
    });
  }

  Future<void> _autoCropEcgRoi() async {
    if (_imageBytes == null) return;

    _showSnack('Auto-crop coming next (OpenCV endpoint).');
    // later:
    // final cropped = await EcgUploadService.autoCropEcg(...);
    // setState(() => _imageBytes = cropped);
  }

  /* -------------------- UPLOAD -------------------- */

  Future<void> _uploadToServer() async {
    if (_imageBytes == null) {
      _showSnack('Please scan or import an ECG first.');
      return;
    }

    setState(() => _isUploading = true);

    try {
      final fileName = _fileName ?? 'ecg.jpg';

      final streamedResp = await EcgUploadService.uploadEcg(
        imageBytes: _imageBytes!,
        fileName: fileName,
        patientName: widget.patientName,
        patientId: widget.patientId,
      );

      final body = await streamedResp.stream.bytesToString();

      if (streamedResp.statusCode >= 200 && streamedResp.statusCode < 300) {
        if (mounted) _showSnack('✅ Uploaded!');
        if (mounted) Navigator.pop(context);
      } else {
        if (mounted) {
          _showSnack('❌ Upload failed [${streamedResp.statusCode}]: $body');
        }
      }
    } catch (e) {
      if (mounted) _showSnack('Error uploading: $e');
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  /* -------------------- UI BUILD -------------------- */

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null;

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FC),
      body: CustomScrollView(
        slivers: [
          _buildHeader(context),
          _buildActionRow(),
          _buildPatientCard(),
          if (hasImage) _buildPreview(),
          _buildRecentSection(),
        ],
      ),
      bottomNavigationBar: _buildBottomBar(hasImage),
    );
  }

  SliverAppBar _buildHeader(BuildContext context) {
    final theme = Theme.of(context);

    return SliverAppBar(
      pinned: true,
      expandedHeight: 220,
      backgroundColor: const Color(0xFF0E6B6F),
      foregroundColor: Colors.white,
      title: const Text('ECG Intake'),
      flexibleSpace: FlexibleSpaceBar(
        background: Stack(
          fit: StackFit.expand,
          children: [
            const DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFF0E6B6F), Color(0xFF073A40)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
              ),
            ),
            const GridHeaderPainter(opacity: .05),
            const ECGWaveHeaderPainter(color: Colors.white70),
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
    );
  }

  SliverToBoxAdapter _buildActionRow() {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
        child: Row(
          children: [
            Expanded(
              child: ActionCard(
                title: 'Scan / Camera',
                subtitle: 'Mobile: scanner • Web: camera • Desktop: file',
                icon: Icons.camera_alt_rounded,
                color: const Color(0xFF1C7C80),
                onTap: _scanWithCamera,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ActionCard(
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
    );
  }

  SliverToBoxAdapter _buildPatientCard() {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 16,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: const Color(0xFFEEF6FF),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.person_outline),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.patientName,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'ID: ${widget.patientId}',
                      style: TextStyle(
                        color: Colors.black.withValues(alpha: 0.60),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  SliverToBoxAdapter _buildPreview() {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: PreviewCard(
          onCrop: _manualCrop,
          onAutoTrim: _autoCropEcgRoi,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: Image.memory(_imageBytes!, fit: BoxFit.cover),
          ),
        ),
      ),
    );
  }

  SliverToBoxAdapter _buildRecentSection() {
    final theme = Theme.of(context);

    return SliverToBoxAdapter(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 10, 16, 0),
            child: Row(
              children: [
                Text(
                  'Recent ECGs',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                if (_recent.isNotEmpty)
                  TextButton(
                    onPressed: () {},
                    child: const Text('View all'),
                  ),
              ],
            ),
          ),
          if (_recent.isEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.history, color: Colors.black45),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'No recent ECGs in this session.',
                        style: TextStyle(
                          color: Colors.black.withValues(alpha: 0.65),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          else
            SizedBox(
              height: 120,
              child: ListView.separated(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                scrollDirection: Axis.horizontal,
                itemCount: _recent.length,
                separatorBuilder: (_, __) => const SizedBox(width: 12),
                itemBuilder: (_, i) {
                  return RecentThumb(
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
          const SizedBox(height: 120),
        ],
      ),
    );
  }

  Widget _buildBottomBar(bool hasImage) {
    return SafeArea(
      minimum: const EdgeInsets.fromLTRB(16, 8, 16, 16),
      child: ElevatedButton.icon(
        icon: _isUploading
            ? const SizedBox(
          width: 18,
          height: 18,
          child: CircularProgressIndicator(strokeWidth: 2),
        )
            : const Icon(Icons.cloud_upload_rounded),
        label: const Padding(
          padding: EdgeInsets.symmetric(vertical: 14),
          child: Text(
            'Upload to Server',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
        ),
        onPressed: (!_isUploading && hasImage) ? _uploadToServer : null,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF0E6B6F),
          disabledBackgroundColor: const Color(0xFF9FB9BC),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
    );
  }

  /* -------------------- HELPERS -------------------- */

  void _rememberRecent(ImageProvider provider) {
    _recent.insert(0, provider);
    while (_recent.length > 12) {
      _recent.removeLast();
    }
  }

  Future<File> _bytesToTempFile(
      Uint8List bytes, {
        required String fileName,
      }) async {
    final dir = await getTemporaryDirectory();
    final file = File(
      p.join(
        dir.path,
        'ecg_${DateTime.now().millisecondsSinceEpoch}_$fileName',
      ),
    );
    await file.writeAsBytes(bytes, flush: true);
    return file;
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }
}