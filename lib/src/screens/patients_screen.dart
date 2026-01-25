import 'package:flutter/material.dart';
import '../services/patient_service.dart';

enum Gender { male, female }

class PatientsScreen extends StatefulWidget {
  const PatientsScreen({super.key});

  @override
  State<PatientsScreen> createState() => _PatientsScreenState();
}

class _PatientsScreenState extends State<PatientsScreen> {
  final nameCtrl = TextEditingController();
  final idCtrl = TextEditingController();
  Gender? gender;

  bool creating = false;

  List<Map<String, dynamic>> patients = [];
  bool loading = true;
  String? loadError;

  @override
  void initState() {
    super.initState();
    _loadPatients();
  }

  @override
  void dispose() {
    nameCtrl.dispose();
    idCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadPatients() async {
    setState(() {
      loading = true;
      loadError = null;
    });

    try {
      final items = await PatientService.fetchPatientsRaw();
      setState(() {
        patients = items;
        loading = false;
      });
    } catch (e) {
      setState(() {
        loadError = e.toString();
        loading = false;
      });
    }
  }

  String? _toBackendGender() {
    if (gender == null) return null;
    return gender == Gender.male ? 'M' : 'F';
  }

  String _genderLabel(String? g) {
    if (g == 'M') return 'Male';
    if (g == 'F') return 'Female';
    return '';
  }

  Future<void> createPatient() async {
    final name = nameCtrl.text.trim();
    final id = idCtrl.text.trim();

    if (name.isEmpty || id.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter patient name and ID')),
      );
      return;
    }

    // optional client-side check (backend is the real enforcement)
    final idExistsLocal = patients.any((p) => (p['patient_id']?.toString() ?? '') == id);
    if (idExistsLocal) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Patient ID already exists')),
      );
      return;
    }

    setState(() => creating = true);

    try {
      await PatientService.createPatient(
        patientId: id,
        fullName: name,
        gender: _toBackendGender(),
      );

      nameCtrl.clear();
      idCtrl.clear();
      setState(() => gender = null);

      await _loadPatients();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Patient created successfully')),
        );
      }
    } catch (e) {
      if (mounted) {
        final msg = e.toString().replaceAll('Exception:', '').trim();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(msg.isEmpty ? 'Create failed' : msg)),
        );
      }
    } finally {
      if (mounted) setState(() => creating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            children: [
              Text(
                'Patients',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
              ),
              const Spacer(),
              IconButton(
                tooltip: 'Refresh',
                onPressed: _loadPatients,
                icon: const Icon(Icons.refresh),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Create Card
          Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Create Patient',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: nameCtrl,
                          decoration: const InputDecoration(
                            labelText: 'Full Name',
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: TextField(
                          controller: idCtrl,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(
                            labelText: 'Patient ID (unique)',
                            border: OutlineInputBorder(),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  Wrap(
                    spacing: 12,
                    runSpacing: 10,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      Text(
                        'Gender:',
                        style: TextStyle(
                          color: cs.onSurface.withValues(alpha: 0.75),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      ChoiceChip(
                        label: const Text('Male'),
                        selected: gender == Gender.male,
                        onSelected: creating ? null : (_) => setState(() => gender = Gender.male),
                      ),
                      ChoiceChip(
                        label: const Text('Female'),
                        selected: gender == Gender.female,
                        onSelected:
                        creating ? null : (_) => setState(() => gender = Gender.female),
                      ),
                      FilledButton.icon(
                        onPressed: creating ? null : createPatient,
                        icon: creating
                            ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                            : const Icon(Icons.add),
                        label: Text(creating ? 'Creating...' : 'Create'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),
          Text(
            'Existing Patients',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),

          // List
          Expanded(
            child: loading
                ? const Center(child: CircularProgressIndicator())
                : loadError != null
                ? Center(
              child: Text(
                'Failed to load patients.\n$loadError',
                textAlign: TextAlign.center,
              ),
            )
                : patients.isEmpty
                ? const Center(child: Text('No patients yet. Create one above.'))
                : ListView.separated(
              itemCount: patients.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (_, i) {
                final p = patients[i];
                final name = (p['full_name'] ?? '').toString();
                final id = (p['patient_id'] ?? '').toString();
                final g = p['gender']?.toString();
                final ecgCount = (p['ecg_count'] ?? 0).toString();

                final gLabel = _genderLabel(g);
                final subtitle = [
                  'ID: $id',
                  if (gLabel.isNotEmpty) gLabel,
                  'ECGs: $ecgCount',
                ].join(' • ');

                return ListTile(
                  leading: CircleAvatar(
                    child: Text(
                      name.isNotEmpty ? name[0].toUpperCase() : '?',
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ),
                  title: Text(name),
                  subtitle: Text(subtitle),
                  onTap: () {
                    Navigator.of(context).pushNamed(
                      '/patient',
                      arguments: {
                        'patient_id': id,
                        'name': name,
                        'gender': g,
                      },
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}