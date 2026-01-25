import 'package:flutter/material.dart';
import 'src/widgets/app_router.dart';

void main() => runApp(const ECGApp());

class ECGApp extends StatelessWidget {
  const ECGApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ECG Analyzer',

      // ✅ use router (fixes "unused import" + app starts at AppShell)
      onGenerateRoute: AppRouter.onGenerateRoute,
      initialRoute: AppRouter.shell,

      theme: ThemeData(
        useMaterial3: true,

        // ✅ remove deprecated background:
        colorScheme: const ColorScheme.light(
          primary: Color(0xFF0F62FE),
          secondary: Color(0xFF00B4A6),
          surface: Color(0xFFFFFFFF),
          onPrimary: Colors.white,
          onSurface: Color(0xFF0B1220),
        ),

        // ✅ use scaffoldBackgroundColor instead of colorScheme.background
        scaffoldBackgroundColor: const Color(0xFFF0F4FA),

        appBarTheme: const AppBarTheme(
          elevation: 0,
          backgroundColor: Colors.transparent,
          foregroundColor: Color(0xFF0B1220),
        ),

        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
        ),

        navigationRailTheme: const NavigationRailThemeData(
          backgroundColor: Color(0xFFF8FAFF),
          indicatorColor: Color(0x1A0F62FE),
          selectedIconTheme: IconThemeData(color: Color(0xFF0F62FE)),
          selectedLabelTextStyle: TextStyle(
            color: Color(0xFF0F62FE),
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}