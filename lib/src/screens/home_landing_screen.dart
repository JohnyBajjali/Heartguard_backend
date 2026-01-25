import 'package:flutter/material.dart';

class HomeLandingScreen extends StatelessWidget {
  const HomeLandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return SingleChildScrollView(
      child: Stack(
        children: [
          // Background shapes (no images)
          Positioned(
            top: -120,
            left: -120,
            child: _SoftCircle(color: cs.primary.withValues(alpha: 0.12), size: 320),
          ),
          Positioned(
            top: 40,
            right: -140,
            child: _SoftCircle(color: cs.secondary.withValues(alpha:0.10), size: 360),
          ),
          Positioned(
            bottom: -160,
            left: 80,
            child: _SoftCircle(color: cs.tertiary.withValues(alpha:0.10), size: 380),
          ),

          Padding(
            padding: const EdgeInsets.all(24),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1150),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // HERO
                    Container(
                      padding: const EdgeInsets.all(22),
                      decoration: const BoxDecoration(
                        borderRadius: BorderRadius.all(Radius.circular(24)),
                        gradient: LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: [
                            Color(0xFF0F62FE), // Medical Blue
                            Color(0xFF00B4A6), // Teal
                          ],
                        ),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Icon badge
                          Container(
                            width: 56,
                            height: 56,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(16),
                              color: Colors.white.withValues(alpha: 0.18),
                            ),
                            child: const Icon(
                              Icons.monitor_heart,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(width: 16),

                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'ECG Analyzer',
                                  style: const TextStyle(
                                    fontSize: 32,
                                    fontWeight: FontWeight.w800,
                                    color: Colors.white,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'A modern dashboard to manage patients, upload ECG documents, and explore 12-lead ECG examples.',
                                  style: TextStyle(
                                    fontSize: 15.5,
                                    height: 1.35,
                                    color: Colors.white.withValues(alpha: 0.9),
                                  ),
                                ),
                                const SizedBox(height: 16),

                                Wrap(
                                  spacing: 10,
                                  runSpacing: 10,
                                  children: const [
                                    _Pill(icon: Icons.people, text: 'Patients'),
                                    _Pill(icon: Icons.upload_file, text: 'ECG Upload'),
                                    _Pill(icon: Icons.menu_book, text: '12-Lead Gallery'),
                                  ],
                                ),

                                const SizedBox(height: 18),

                                // ECG Wave visualization (no images)
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(16),
                                  child: Container(
                                    height: 90,
                                    width: double.infinity,
                                    color: Colors.white.withValues(alpha:0.75),
                                    child: CustomPaint(
                                      painter: _EcgWavePainter(),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 18),

                    // Feature cards
                    LayoutBuilder(
                      builder: (context, c) {
                        final wide = c.maxWidth >= 950;
                        final w = wide ? (c.maxWidth - 32) / 3 : c.maxWidth;

                        return Wrap(
                          spacing: 16,
                          runSpacing: 16,
                          children: [
                            _FeatureCard(
                              width: w,
                              icon: Icons.people,
                              title: 'Patients',
                              text:
                              'Browse existing patients or create new ones using a unique Patient ID.',
                            ),
                            _FeatureCard(
                              width: w,
                              icon: Icons.document_scanner,
                              title: 'Scan & Upload',
                              text:
                              'Capture ECG images, crop them, and upload them to your backend.',
                            ),
                            _FeatureCard(
                              width: w,
                              icon: Icons.monitor_heart,
                              title: '12-Lead Examples',
                              text:
                              'Open curated 12-lead ECG examples from the web for learning and comparison.',
                            ),
                          ],
                        );
                      },
                    ),

                    const SizedBox(height: 18),

                    // About section
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha:0.90),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(color: Colors.black.withValues(alpha:0.06)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'About ECG Analysis',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                              color: cs.onSurface,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'An ECG (electrocardiogram) records the electrical activity of the heart. '
                                'This application helps you organize patient ECG uploads and provides reference 12-lead examples.',
                            style: TextStyle(
                              height: 1.4,
                              color: cs.onSurface.withValues(alpha:0.75),
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 18),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SoftCircle extends StatelessWidget {
  final Color color;
  final double size;
  const _SoftCircle({required this.color, required this.size});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  final IconData icon;
  final String text;
  const _Pill({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha:0.75),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.black.withValues(alpha:0.06)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: cs.primary),
          const SizedBox(width: 8),
          Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final double width;
  final IconData icon;
  final String title;
  final String text;

  const _FeatureCard({
    required this.width,
    required this.icon,
    required this.title,
    required this.text,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return SizedBox(
      width: width,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha:0.92),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: Colors.black.withValues(alpha:0.06)),
        ),
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                color: cs.primary.withValues(alpha: 0.12),
              ),
              child: Icon(icon, color: cs.primary),
            ),
            const SizedBox(height: 12),
            Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text(text, style: const TextStyle(height: 1.35)),
          ],
        ),
      ),
    );
  }
}

class _EcgWavePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final gridPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    // light grid
    gridPaint.color = Colors.black.withValues(alpha:0.06);
    const step = 18.0;
    for (double x = 0; x <= size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }
    for (double y = 0; y <= size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // ECG wave
    final wavePaint = Paint()
      ..color = Colors.black.withValues(alpha:0.65)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.2
      ..strokeCap = StrokeCap.round;

    final path = Path();
    final midY = size.height * 0.55;
    path.moveTo(0, midY);

    // Simple repeating ECG-like pattern
    double x = 0;
    while (x < size.width) {
      path.lineTo(x + 10, midY);
      path.lineTo(x + 18, midY - 6);
      path.lineTo(x + 24, midY + 10);
      path.lineTo(x + 28, midY - 26); // spike up
      path.lineTo(x + 34, midY + 18); // drop
      path.lineTo(x + 44, midY);
      path.lineTo(x + 62, midY);
      x += 68;
    }

    canvas.drawPath(path, wavePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}