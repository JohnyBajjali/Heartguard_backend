import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:google_mlkit_document_scanner/google_mlkit_document_scanner.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../services/ecg_upload_service.dart';
import '../services/ecg_placeholder_service.dart';
import '../widgets/action_card.dart';
import '../widgets/preview_card.dart';
import '../widgets/recent_thumb.dart';
import '../widgets/step_chip.dart';
import '../painters/ecg_header_painters.dart';

class CaptureUploadScreen extends StatefulWidget {
  const CaptureUploadScreen({super.key});

  @override
  State<CaptureUploadScreen> createState() => _CaptureUploadScreenState();
}

class _CaptureUploadScreenState extends State<CaptureUploadScreen> {
  Uint8List? _imageBytes;
  String? _fileName;
  String? _imagePath;
  bool _isUploading = false;

  final List<ImageProvider> _recent = [];
  final TextEditingController _patientNameController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadSamplePlaceholders();
  }

  @override
  void dispose() {
    _patientNameController.dispose();
    super.dispose();
  }

  /* -------------------- INIT PLACEHOLDERS -------------------- */

  Future<void> _loadSamplePlaceholders() async {
    final bytes = await EcgPlaceholderService.generateSampleEcgPngBytes();
    if (!mounted) return;
    setState(() {
      _recent
        ..clear()
        ..addAll(List.generate(4, (_) => MemoryImage(bytes)));
    });
  }

  /* -------------------- IMAGE PICK / SCAN -------------------- */

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source);
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(picked.path);
      _imagePath = picked.path;
      _rememberRecent(MemoryImage(bytes));
    });
  }

  Future<void> _scanWithCamera() async {
    if (kIsWeb) {
      await _pickImage(ImageSource.camera);
      return;
    }

    final scanner = DocumentScanner(
      options: const DocumentScannerOptions(
        documentFormat: DocumentFormat.jpeg,
        mode: ScannerMode.filter,
        pageLimit: 1,
        isGalleryImport: false,
      ),
    );

    try {
      final result = await scanner.scanDocument();
      if (!mounted || result == null) return;

      if (result.images.isNotEmpty) {
        final firstPath = result.images.first;
        final bytes = await File(firstPath).readAsBytes();
        setState(() {
          _imageBytes = bytes;
          _fileName = p.basename(firstPath);
          _imagePath = firstPath;
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

  Future<void> _importFromGallery() => _pickImage(ImageSource.gallery);

  /* -------------------- CROP -------------------- */

  Future<void> _manualCrop() async {
    if (_imageBytes == null) return;

    String sourcePath;
    if (kIsWeb) {
      if (_imagePath == null) {
        _showSnack('Please pick or scan an image again before cropping.');
        return;
      }
      sourcePath = _imagePath!;
    } else {
      final temp = await _bytesToTempFile(
        _imageBytes!,
        fileName: _fileName ?? 'ecg.png',
      );
      sourcePath = temp.path;
    }

    final cropped = await ImageCropper().cropImage(
      sourcePath: sourcePath,
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: 'Crop ECG Area',
          hideBottomControls: false,
          lockAspectRatio: false,
          showCropGrid: true,
        ),
        const IOSUiSettings(title: 'Crop ECG Area'),
        WebUiSettings(context: context),
      ],
    );

    if (cropped == null) return;

    final bytes = await cropped.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _fileName = p.basename(cropped.path);
      _imagePath = cropped.path;
      _rememberRecent(MemoryImage(bytes));
    });
  }

  Future<void> _autoCropEcgRoi() async {
    if (_imageBytes == null) return;
    _showSnack('Auto-crop coming soon — use manual crop for now.');
  }

  /* -------------------- UPLOAD -------------------- */

  Future<void> _uploadToServer() async {
    if (_imageBytes == null) {
      _showSnack('Please scan or import an ECG first.');
      return;
    }

    final patientName = _patientNameController.text.trim();
    if (patientName.isEmpty) {
      _showSnack('Please enter the patient name.');
      return;
    }

    setState(() => _isUploading = true);

    try {
      final fileName = _fileName ?? 'ecg.jpg';

      final streamedResp = await EcgUploadService.uploadEcg(
        imageBytes: _imageBytes!,
        fileName: fileName,
        patientName: patientName,
      );

      final body = await streamedResp.stream.bytesToString();

      if (streamedResp.statusCode >= 200 &&
          streamedResp.statusCode < 300) {
        _showSnack('✅ Uploaded! Server response: $body');
      } else {
        _showSnack(
          '❌ Upload failed [${streamedResp.statusCode}]: $body',
        );
      }
    } catch (e) {
      _showSnack('Error uploading: $e');
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
          _buildPatientField(),
          if (hasImage) _buildPreview(),
          _buildRecentSection(),
          _buildPipelineChips(),
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
                subtitle: 'Take a photo of the paper',
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

  SliverToBoxAdapter _buildPatientField() {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
        child: TextField(
          controller: _patientNameController,
          decoration: const InputDecoration(
            labelText: 'Patient Name',
            hintText: 'Enter patient name',
            border: OutlineInputBorder(),
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
          child: ClipRRect(
            borderRadius: BorderRadius.circular(18),
            child: Image.memory(_imageBytes!, fit: BoxFit.cover),
          ),
          onCrop: _manualCrop,
          onAutoTrim: _autoCropEcgRoi,
        ),
      ),
    );
  }

  SliverToBoxAdapter _buildRecentSection() {
    return SliverToBoxAdapter(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: Row(
              children: [
                Text(
                  'Recent ECGs',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                TextButton(
                  onPressed: () {},
                  child: const Text('View all'),
                ),
              ],
            ),
          ),
          SizedBox(
            height: 120,
            child: ListView.separated(
              padding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
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
        ],
      ),
    );
  }

  SliverToBoxAdapter _buildPipelineChips() {
    return SliverToBoxAdapter(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 100),
        child: Wrap(
          spacing: 8,
          runSpacing: -6,
          children: const [
            StepChip(label: 'Auto-crop grid'),
            StepChip(label: 'Deskew & de-shadow'),
            StepChip(label: 'Digitize waveform'),
            StepChip(label: 'Send to AI model'),
          ],
        ),
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
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }
}
