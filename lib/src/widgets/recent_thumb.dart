import 'package:flutter/material.dart';

class RecentThumb extends StatelessWidget {
  const RecentThumb({
    super.key,
    required this.image,
    required this.onTap,
  });

  final ImageProvider image;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 160,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.black12),
          image: DecorationImage(
            image: image,
            fit: BoxFit.cover,
          ),
        ),
      ),
    );
  }
}
