import 'package:flutter/material.dart';

class GridHeaderPainter extends StatelessWidget {
  const GridHeaderPainter({super.key, this.opacity = .07});

  final double opacity;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _GridPainterImpl(opacity),
      size: Size.infinite,
    );
  }
}

class _GridPainterImpl extends CustomPainter {
  _GridPainterImpl(this.opacity);

  final double opacity;

  @override
  void paint(Canvas canvas, Size size) {
    final big = Paint()
      ..color = Colors.white.withOpacity(opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final small = Paint()
      ..color = Colors.white.withOpacity(opacity * .4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = .5;

    const bigStep = 40.0;
    const smallStep = 8.0;

    for (double x = 0; x <= size.width; x += smallStep) {
      canvas.drawLine(
        Offset(x, 0),
        Offset(x, size.height),
        x % bigStep == 0 ? big : small,
      );
    }
    for (double y = 0; y <= size.height; y += smallStep) {
      canvas.drawLine(
        Offset(0, y),
        Offset(size.width, y),
        y % bigStep == 0 ? big : small,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class ECGWaveHeaderPainter extends StatelessWidget {
  const ECGWaveHeaderPainter({
    super.key,
    required this.color,
  });

  final Color color;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _ECGWavePainterImpl(color),
      size: Size.infinite,
    );
  }
}

class _ECGWavePainterImpl extends CustomPainter {
  _ECGWavePainterImpl(this.color);

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final p = Path();
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final mid = size.height * .55;
    final w = size.width;

    p.moveTo(-40, mid);
    for (double x = -40; x < w + 40; x += 160) {
      p.quadraticBezierTo(x + 20, mid - 8, x + 40, mid);
      p.lineTo(x + 55, mid + 4);
      p.lineTo(x + 68, mid - 42);
      p.lineTo(x + 82, mid + 10);
      p.lineTo(x + 95, mid - 6);
      p.quadraticBezierTo(x + 120, mid - 2, x + 140, mid - 4);
      p.quadraticBezierTo(x + 160, mid + 8, x + 180, mid);
    }

    // Thin line
    canvas.drawPath(p, paint);

    // Glow
    canvas.drawPath(
      p,
      paint
        ..color = color.withOpacity(.25)
        ..strokeWidth = 6,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
