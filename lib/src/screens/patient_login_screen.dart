import 'package:flutter/material.dart';

import '../services/patient_service.dart';
import 'capture_upload_screen.dart';

class PatientLoginScreen extends StatefulWidget {
  const PatientLoginScreen({super.key});

  @override
  State<PatientLoginScreen> createState() => _PatientLoginScreenState();
}

class _PatientLoginScreenState extends State<PatientLoginScreen> {
  final _nameController = TextEditingController();
  final _idController = TextEditingController();

  bool _isLoading = false;
  String? _error;

  @override
  void dispose() {
    _nameController.dispose();
    _idController.dispose();
    super.dispose();
  }

  Future<void> _onContinuePressed() async {
    final name = _nameController.text.trim();
    final id = _idController.text.trim();

    if (name.isEmpty) {
      setState(() => _error = 'Please enter patient name.');
      return;
    }
    if (id.isEmpty) {
      setState(() => _error = 'Please enter patient ID.');
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final patient = await PatientService.loginOrCreate(
        patientId: id,
        name: name,
      );

      if (!mounted) return;

      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => CaptureUploadScreen(
            patientId: patient.id,
            patientName: patient.name,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      final msg = e.toString().replaceFirst('Exception: ', '');
      setState(() => _error = msg);
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFFF4F8FF),
              Color(0xFFF7F9FC),
              Color(0xFFFFFFFF),
            ],
          ),
        ),
        child: Stack(
          children: [
            // Background blobs
            const Positioned(
              top: -120,
              left: -120,
              child: _BlurBlob(
                size: 280,
                color: Color(0xFF3B82F6),
                opacity: 0.12,
              ),
            ),
            const Positioned(
              bottom: -140,
              right: -140,
              child: _BlurBlob(
                size: 320,
                color: Color(0xFF10B981),
                opacity: 0.10,
              ),
            ),
            const Positioned(
              top: 80,
              right: -90,
              child: _BlurBlob(
                size: 220,
                color: Color(0xFF6366F1),
                opacity: 0.10,
              ),
            ),

            // Main content
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.92),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.08),
                          blurRadius: 30,
                          offset: const Offset(0, 12),
                        ),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.favorite,
                          size: 60,
                          color: theme.colorScheme.primary,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'Patient Login',
                          style: theme.textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 22),

                        // Name FIRST
                        TextField(
                          controller: _nameController,
                          decoration: InputDecoration(
                            labelText: 'Patient Name',
                            filled: true,
                            fillColor: const Color(0xFFF8FAFF),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          textInputAction: TextInputAction.next,
                        ),
                        const SizedBox(height: 12),

                        // ID SECOND
                        TextField(
                          controller: _idController,
                          decoration: InputDecoration(
                            labelText: 'Patient ID',
                            filled: true,
                            fillColor: const Color(0xFFF8FAFF),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          keyboardType: TextInputType.number,
                          textInputAction: TextInputAction.done,
                          onSubmitted: (_) =>
                          _isLoading ? null : _onContinuePressed(),
                        ),

                        const SizedBox(height: 16),

                        if (_error != null) ...[
                          Text(
                            _error!,
                            style: const TextStyle(color: Colors.red),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 8),
                        ],

                        SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            onPressed:
                            _isLoading ? null : _onContinuePressed,
                            child: _isLoading
                                ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                              ),
                            )
                                : const Text('Continue'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/* ---------- Background Blob Widget ---------- */
class _BlurBlob extends StatelessWidget {
  final double size;
  final Color color;
  final double opacity;

  const _BlurBlob({
    required this.size,
    required this.color,
    required this.opacity,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color.withOpacity(opacity),
        shape: BoxShape.circle,
      ),
    );
  }
}
