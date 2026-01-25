import 'package:flutter/material.dart';

class PreviewCard extends StatelessWidget {
  const PreviewCard({
    super.key,
    required this.child,
    required this.onCrop,
    required this.onAutoTrim,
  });

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
          border: Border.all(
            color: theme.colorScheme.primary.withOpacity(.18),
          ),
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
