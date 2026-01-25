import 'package:flutter/material.dart';

class StepChip extends StatelessWidget {
  const StepChip({
    super.key,
    required this.label,
  });

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
