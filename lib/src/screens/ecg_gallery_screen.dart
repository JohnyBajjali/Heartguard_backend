import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart' as launcher;

class EcgGalleryScreen extends StatefulWidget {
  const EcgGalleryScreen({super.key});

  @override
  State<EcgGalleryScreen> createState() => _EcgGalleryScreenState();
}

class _EcgGalleryScreenState extends State<EcgGalleryScreen> {
  final _searchCtrl = TextEditingController();
  String _category = 'All';

  // ✅ Curated 12-lead ECG learning sources (reliable + educational)
  // You can add more links later easily.
  final List<_EcgWebItem> _items = const [
    _EcgWebItem(
      title: 'Normal Sinus Rhythm',
      category: 'Normal',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/normal-sinus-rhythm-ecg-library/',
      tags: ['normal', 'sinus', 'rhythm'],
    ),
    _EcgWebItem(
      title: 'Atrial Fibrillation (AF)',
      category: 'Arrhythmia',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/atrial-fibrillation-ecg-library/',
      tags: ['af', 'atrial fibrillation', 'irregular'],
    ),
    _EcgWebItem(
      title: 'Atrial Flutter',
      category: 'Arrhythmia',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/atrial-flutter-ecg-library/',
      tags: ['flutter', 'sawtooth'],
    ),
    _EcgWebItem(
      title: 'Ventricular Tachycardia (VT)',
      category: 'Arrhythmia',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/ventricular-tachycardia-ecg-library/',
      tags: ['vt', 'wide qrs'],
    ),
    _EcgWebItem(
      title: 'Acute STEMI (ST Elevation MI)',
      category: 'Ischemia / MI',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/st-elevation-mi-ecg-library/',
      tags: ['stemi', 'mi', 'st elevation'],
    ),
    _EcgWebItem(
      title: 'Inferior STEMI',
      category: 'Ischemia / MI',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/inferior-stemi-ecg-library/',
      tags: ['inferior', 'stemi', 'ii iii avf'],
    ),
    _EcgWebItem(
      title: 'Anterior STEMI',
      category: 'Ischemia / MI',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/anterior-stemi-ecg-library/',
      tags: ['anterior', 'stemi', 'v1-v4'],
    ),
    _EcgWebItem(
      title: 'Left Bundle Branch Block (LBBB)',
      category: 'Conduction',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/',
      tags: ['lbbb', 'conduction'],
    ),
    _EcgWebItem(
      title: 'Right Bundle Branch Block (RBBB)',
      category: 'Conduction',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/',
      tags: ['rbbb', 'rsr'],
    ),
    _EcgWebItem(
      title: 'Left Ventricular Hypertrophy (LVH)',
      category: 'Hypertrophy',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/left-ventricular-hypertrophy-lvh-ecg-library/',
      tags: ['lvh', 'voltage'],
    ),
    _EcgWebItem(
      title: 'Hyperkalemia',
      category: 'Electrolytes',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/hyperkalaemia-ecg-library/',
      tags: ['hyperkalemia', 't waves'],
    ),
    _EcgWebItem(
      title: 'Pericarditis',
      category: 'Other',
      source: 'LITFL ECG Library',
      url: 'https://litfl.com/pericarditis-ecg-library/',
      tags: ['pericarditis', 'diffuse st elevation'],
    ),
  ];

  List<String> get _categories {
    final s = <String>{'All'};
    for (final i in _items) {
      s.add(i.category);
    }
    return s.toList()..sort((a, b) => a == 'All' ? -1 : a.compareTo(b));
  }

  List<_EcgWebItem> get _filtered {
    final q = _searchCtrl.text.trim().toLowerCase();
    return _items.where((it) {
      final matchCategory = _category == 'All' || it.category == _category;
      if (!matchCategory) return false;

      if (q.isEmpty) return true;

      final hay = [
        it.title,
        it.category,
        it.source,
        ...it.tags,
      ].join(' ').toLowerCase();

      return hay.contains(q);
    }).toList();
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _open(String url) async {
    final uri = Uri.parse(url);

    if (await launcher.canLaunchUrl(uri)) {
      await launcher.launchUrl(uri);
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not open link')),
        );
      }
    }
  }

  IconData _iconForCategory(String c) {
    switch (c) {
      case 'Normal':
        return Icons.favorite_outline;
      case 'Arrhythmia':
        return Icons.graphic_eq;
      case 'Ischemia / MI':
        return Icons.warning_amber_rounded;
      case 'Conduction':
        return Icons.route;
      case 'Hypertrophy':
        return Icons.fitness_center;
      case 'Electrolytes':
        return Icons.science_outlined;
      default:
        return Icons.description_outlined;
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final list = _filtered;

    return Padding(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Title
          Row(
            children: [
              Text(
                '12-Lead ECG Gallery',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
              ),
              const Spacer(),
              Text(
                '${list.length} items',
                style: TextStyle(
                  color: cs.onSurface.withValues(alpha: 0.65),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),

          // Search
          TextField(
            controller: _searchCtrl,
            onChanged: (_) => setState(() {}),
            decoration: const InputDecoration(
              prefixIcon: Icon(Icons.search),
              hintText: 'Search (e.g., STEMI, AFib, LBBB...)',
            ),
          ),
          const SizedBox(height: 10),

          // Categories chips
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: _categories.map((c) {
                final selected = _category == c;
                return Padding(
                  padding: const EdgeInsets.only(right: 10),
                  child: ChoiceChip(
                    label: Text(c),
                    selected: selected,
                    onSelected: (_) => setState(() => _category = c),
                  ),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 14),

          // Content
          Expanded(
            child: list.isEmpty
                ? Center(
              child: Text(
                'No results.\nTry a different search or category.',
                textAlign: TextAlign.center,
                style: TextStyle(color: cs.onSurface.withValues(alpha: 0.7)),
              ),
            )
                : GridView.builder(
              gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 420,
                mainAxisExtent: 170,
                crossAxisSpacing: 14,
                mainAxisSpacing: 14,
              ),
              itemCount: list.length,
              itemBuilder: (context, i) {
                final it = list[i];
                return Card(
                  child: Padding(
                    padding: const EdgeInsets.all(14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(14),
                                color: cs.primary.withValues(alpha: 0.10),
                              ),
                              child: Icon(
                                _iconForCategory(it.category),
                                color: cs.primary,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                it.title,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w800,
                                  fontSize: 15.5,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),

                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            _Badge(text: it.category),
                            _Badge(text: it.source),
                          ],
                        ),
                        const Spacer(),

                        Row(
                          children: [
                            Text(
                              'Open reference',
                              style: TextStyle(
                                color: cs.onSurface.withValues(alpha: 0.7),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const Spacer(),
                            FilledButton.icon(
                              onPressed: () => _open(it.url),
                              icon: const Icon(Icons.open_in_new),
                              label: const Text('Open'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),

          const SizedBox(height: 8),
          Text(
            'Note: These are web references for learning/comparison. Your uploaded ECGs remain in the Upload workflow.',
            style: TextStyle(
              color: cs.onSurface.withValues(alpha: 0.6),
              fontSize: 12.5,
            ),
          ),
        ],
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final String text;
  const _Badge({required this.text});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(999),
        color: cs.surface,
        border: Border.all(
          color: cs.onSurface.withValues(alpha: 0.10),
        ),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: cs.onSurface.withValues(alpha: 0.78),
        ),
      ),
    );
  }
}

class _EcgWebItem {
  final String title;
  final String category;
  final String source;
  final String url;
  final List<String> tags;

  const _EcgWebItem({
    required this.title,
    required this.category,
    required this.source,
    required this.url,
    this.tags = const [],
  });
}