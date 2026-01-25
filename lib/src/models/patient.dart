// lib/src/models/patient.dart

class EcgRecord {
  final int id;
  final String imageUrl; // can be relative (/media/...) or absolute (http://...)
  final DateTime uploadedAt;

  EcgRecord({
    required this.id,
    required this.imageUrl,
    required this.uploadedAt,
  });

  factory EcgRecord.fromJson(Map<String, dynamic> json) {
    final rawUrl = (json['imageUrl'] ?? json['url'] ?? json['file'] ?? '').toString();

    return EcgRecord(
      id: (json['id'] as num).toInt(),
      imageUrl: rawUrl,
      uploadedAt: DateTime.parse(json['uploaded_at'] as String),
    );
  }
}

class Patient {
  final String id; // patient_id as string (UI uses text fields)
  final String name;
  final String? gender; // "M"/"F"/null
  final List<EcgRecord> ecgs;

  Patient({
    required this.id,
    required this.name,
    required this.ecgs,
    this.gender,
  });

  factory Patient.fromJson(Map<String, dynamic> json) {
    final ecgList = (json['ecgs'] as List<dynamic>? ?? [])
        .map((e) => EcgRecord.fromJson(e as Map<String, dynamic>))
        .toList();

    return Patient(
      id: json['patient_id'].toString(),
      name: (json['name'] ?? json['full_name'] ?? '').toString(),
      gender: json['gender']?.toString(),
      ecgs: ecgList,
    );
  }
}