// lib/src/widgets/ecg_intake_panel.dart
import 'dart:io' show File, Platform;
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:google_mlkit_document_scanner/google_mlkit_document_scanner.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../screens/manual_crop_screen.dart';
import '../services/ecg_upload_service.dart';
import 'action_card.dart';

class EcgIntakePanel extends StatefulWidget {
  final String patientId;
  final String patientName;

  final bool startWithScanner;
  final bool startWithUpload;

  /// Show/hide X close button
  final bool showClose;

  final VoidCallback onUploaded;
  final VoidCallback onClose;

  const EcgIntakePanel({
    super.key,
    required this.patientId,
    required this.patientName,
    this.startWithScanner = false,
    this.startWithUpload = false,
    this.showClose = true,
    required this.onUploaded,
    required this.onClose,
  });

  @override
  State<EcgIntakePanel> createState() => _EcgIntakePanelState();
}

class _EcgIntakePanelState extends State<EcgIntakePanel> {
  Uint8List? _imageBytes;
  String? _fileName;
  String? _imagePath;

  bool _isUploading = false;
  bool _isAutoCropping = false;

  bool _didAutoStart = false;

  @override
  void initState() {
    super.initState();
    _maybeAutoStart();
  }

  @override
  void didUpdateWidget(covariant EcgIntakePanel oldWidget) {
    super.didUpdateWidget(oldWidget);

    final scannerBecameTrue =
        !oldWidget.startWithScanner && widget.startWithScanner;
    final uploadBecameTrue =
        !oldWidget.startWithUpload && widget.startWithUpload;

    if (scannerBecameTrue || uploadBecameTrue) {
      _didAutoStart = false;
      _maybeAutoStart();
    }
  }

  void _maybeAutoStart() {
    if (_didAutoStart) return;
    if (_imageBytes != null) return;

    if (widget.startWithScanner) {
      _didAutoStart = true;
      Future.microtask(() async {
        if (!mounted) return;
        await _scan();
      });
    } else if (widget.startWithUpload) {
      _didAutoStart = true;
      Future.microtask(() async {
        if (!mounted) return;
        await _importFromFiles();
      });
    }
  }

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
    });
  }

  /// ✅ Upload (works everywhere)
  Future<void> _importFromFiles() async {
    await _pickImage(ImageSource.gallery);
  }

  /// ✅ Scan (best per platform)
  Future<void> _scan() async {
    // WEB: browser camera
    if (kIsWeb) {
      await _pickImage(ImageSource.camera);
      return;
    }

    final isMobile = Platform.isAndroid || Platform.isIOS;

    // DESKTOP: use file picker (camera not reliable)
    if (!isMobile) {
      await _pickImage(ImageSource.gallery);
      return;
    }

    // ANDROID/iOS: MLKit document scanner
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
        });
      } else if (result.pdf != null) {
        _showSnack('Scanner returned PDF; configure scanner to return images.');
      }
    } catch (_) {
      if (mounted) _showSnack('Scan cancelled or failed.');
    } finally {
      await scanner.close();
    }
  }

  // ---------- Crop menu ----------
  void _openCropMenu() {
    if (_imageBytes == null) return;

    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (ctx) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 6, 12, 12),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ListTile(
                  leading: const Icon(Icons.auto_fix_high),
                  title: const Text('Automatic crop'),
                  subtitle: const Text('Remove borders & titles automatically'),
                  onTap: () {
                    Navigator.pop(ctx);
                    _autoCropEcgRoi();
                  },
                ),
                const Divider(height: 1),
                ListTile(
                  leading: const Icon(Icons.crop),
                  title: const Text('Manual crop'),
                  subtitle: const Text('Crop the ECG area yourself'),
                  onTap: () {
                    Navigator.pop(ctx);
                    _manualCrop();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _manualCrop() async {
    if (_imageBytes == null) return;

    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    // ✅ Web + Desktop: use crop_your_image screen (resizable + Done button)
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
        // _imagePath not guaranteed (bytes-based crop)
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
    });
  }

  // ---------- Auto crop (backend OpenCV) ----------
  Future<void> _autoCropEcgRoi() async {
    if (_imageBytes == null) return;

    setState(() => _isAutoCropping = true);
    try {
      final cropped = await EcgUploadService.autoCropEcg(
        imageBytes: _imageBytes!,
        fileName: _fileName ?? 'ecg.jpg',
      );

      if (!mounted) return;
      setState(() {
        _imageBytes = cropped;
        _fileName = 'cropped_${_fileName ?? 'ecg.jpg'}';
      });

      _showSnack('✅ Auto crop done');
    } catch (e) {
      if (!mounted) return;
      _showSnack('❌ Auto crop failed: $e');
    } finally {
      if (mounted) setState(() => _isAutoCropping = false);
    }
  }

  Future<void> _uploadToServer() async {
    if (_imageBytes == null) {
      _showSnack('Please scan or upload an ECG first.');
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
      final ok = streamedResp.statusCode >= 200 && streamedResp.statusCode < 300;
      if (!mounted) return;

      if (ok) {
        _showSnack('✅ Uploaded!');
        widget.onUploaded();
        setState(() {
          _imageBytes = null;
          _fileName = null;
          _imagePath = null;
        });
      } else {
        _showSnack('❌ Upload failed [${streamedResp.statusCode}]: $body');
      }
    } catch (e) {
      if (mounted) _showSnack('Error uploading: $e');
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _imageBytes != null;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    'ECG Intake',
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                  const Spacer(),
                  if (widget.showClose)
                    IconButton(
                      tooltip: 'Close',
                      onPressed: widget.onClose,
                      icon: const Icon(Icons.close),
                    ),
                ],
              ),
              const SizedBox(height: 8),

              Row(
                children: [
                  Expanded(
                    child: ActionCard(
                      title: 'Scan / Camera',
                      subtitle: 'Mobile: scanner • Web: camera • Desktop: file',
                      icon: Icons.camera_alt_rounded,
                      color: const Color(0xFF1C7C80),
                      onTap: _scan,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ActionCard(
                      title: 'Upload',
                      subtitle: 'From gallery/files',
                      icon: Icons.folder_open_rounded,
                      color: const Color(0xFF0E6B6F),
                      onTap: _importFromFiles,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              if (hasImage) ...[
                ClipRRect(
                  borderRadius: BorderRadius.circular(14),
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxHeight: 320),
                    child: Container(
                      width: double.infinity,
                      color: Colors.black12,
                      child: Image.memory(_imageBytes!, fit: BoxFit.contain),
                    ),
                  ),
                ),
                const SizedBox(height: 10),

                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    icon: _isAutoCropping
                        ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                        : const Icon(Icons.crop_rounded),
                    label: const Padding(
                      padding: EdgeInsets.symmetric(vertical: 12),
                      child: Text(
                        'Crop',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    onPressed: _isAutoCropping ? null : _openCropMenu,
                  ),
                ),

                const SizedBox(height: 10),

                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    icon: _isUploading
                        ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                        : const Icon(Icons.cloud_upload_rounded),
                    label: const Padding(
                      padding: EdgeInsets.symmetric(vertical: 12),
                      child: Text(
                        'Upload to Server',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    onPressed: _isUploading ? null : _uploadToServer,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0E6B6F),
                      disabledBackgroundColor: const Color(0xFF9FB9BC),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                  ),
                ),
              ] else ...[
                Text(
                  'Choose Scan or Upload to attach an ECG for this patient.',
                  style: TextStyle(color: Colors.black.withValues(alpha: 0.7)),
                ),
              ],
            ],
          ),
        ),
      ),
    );
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