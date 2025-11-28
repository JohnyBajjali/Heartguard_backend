// lib/src/models/patient.dart

class EcgRecord {
  final int id;
  final String imageUrl;
  final DateTime uploadedAt;

  EcgRecord({
    required this.id,
    required this.imageUrl,
    required this.uploadedAt,
  });

  factory EcgRecord.fromJson(Map<String, dynamic> json) {
    return EcgRecord(
      id: json['id'] as int,
      imageUrl: json['file'] as String,           // e.g. "/media/ecgs/123.png"
      uploadedAt: DateTime.parse(json['uploaded_at'] as String),
    );
  }
}

class Patient {
  final String id;        // The patient ID that the user types (string)
  final String name;
  final List<EcgRecord> ecgs;

  Patient({
    required this.id,
    required this.name,
    required this.ecgs,
  });

  factory Patient.fromJson(Map<String, dynamic> json) {
    final ecgList = (json['ecgs'] as List<dynamic>? ?? [])
        .map((e) => EcgRecord.fromJson(e as Map<String, dynamic>))
        .toList();

    return Patient(
      id: json['patient_id'].toString(),
      name: json['name'] as String,
      ecgs: ecgList,
    );
  }
}
