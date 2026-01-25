import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../services/ecg_upload_service.dart';
import '../widgets/ecg_intake_panel.dart';

class PatientDetailScreen extends StatefulWidget {
  final String patientId;
  final String patientName;
  final String? gender;

  const PatientDetailScreen({
    super.key,
    required this.patientId,
    required this.patientName,
    this.gender,
  });

  @override
  State<PatientDetailScreen> createState() => _PatientDetailScreenState();
}

class _PatientDetailScreenState extends State<PatientDetailScreen> {
  bool loading = true;
  String? error;

  List<Map<String, dynamic>> ecgs = [];
  final notesCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadEcgs();
  }

  @override
  void dispose() {
    notesCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadEcgs() async {
    setState(() {
      loading = true;
      error = null;
    });

    try {
      final base = EcgUploadService.backendBaseUrl();
      final uri = Uri.parse('$base/api/ecg/list/?patient_id=${widget.patientId}');
      final res = await http.get(uri);

      if (res.statusCode != 200) {
        throw Exception('Failed to load ECGs (${res.statusCode})');
      }

      final decoded = jsonDecode(res.body);
      final list = (decoded as List).cast<dynamic>();

      setState(() {
        ecgs = list.map((e) => (e as Map).cast<String, dynamic>()).toList();
        loading = false;
      });
    } catch (e) {
      setState(() {
        error = e.toString();
        loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.patientName} (ID: ${widget.patientId})'),
        actions: [
          IconButton(onPressed: _loadEcgs, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ✅ Always show ECG Intake panel
                  EcgIntakePanel(
                    patientId: widget.patientId,
                    patientName: widget.patientName,
                    startWithScanner: false,
                    startWithUpload: false,

                    // ✅ hide X
                    showClose: false,

                    onUploaded: _loadEcgs,

                    // still required, but not used because showClose=false
                    onClose: () {},
                  ),

                  const SizedBox(height: 16),

                  Text('Doctor Notes', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  TextField(
                    controller: notesCtrl,
                    minLines: 3,
                    maxLines: 6,
                    decoration: InputDecoration(
                      hintText: 'Write notes for this patient...',
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),

                  const SizedBox(height: 16),
                  Text('Previous ECGs', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),

                  if (loading)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 24),
                      child: Center(child: CircularProgressIndicator()),
                    )
                  else if (error != null)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 24),
                      child: Center(child: Text(error!, textAlign: TextAlign.center)),
                    )
                  else if (ecgs.isEmpty)
                      Padding(
                        padding: const EdgeInsets.symmetric(vertical: 24),
                        child: Center(
                          child: Text(
                            'No ECGs uploaded for this patient yet.',
                            style: TextStyle(color: cs.onSurface.withValues(alpha: 0.7)),
                          ),
                        ),
                      )
                    else
                      ListView.separated(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: ecgs.length,
                        separatorBuilder: (_, __) => const SizedBox(height: 10),
                        itemBuilder: (_, i) {
                          final e = ecgs[i];
                          final id = e['id']?.toString() ?? '';
                          final created = e['uploaded_at']?.toString() ?? '';
                          final fileUrl = e['url']?.toString() ?? '';

                          return Card(
                            child: ListTile(
                              title: Text('ECG #$id'),
                              subtitle: Text(created.isEmpty ? 'Uploaded' : created),
                              trailing: const Icon(Icons.open_in_new),
                              onTap: fileUrl.isEmpty
                                  ? null
                                  : () {
                                showDialog(
                                  context: context,
                                  builder: (_) => AlertDialog(
                                    title: const Text('ECG URL'),
                                    content: SelectableText(fileUrl),
                                    actions: [
                                      TextButton(
                                        onPressed: () => Navigator.pop(context),
                                        child: const Text('Close'),
                                      ),
                                    ],
                                  ),
                                );
                              },
                            ),
                          );
                        },
                      ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}