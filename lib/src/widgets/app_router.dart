import 'package:flutter/material.dart';

import 'app_shell.dart';
import '../screens/not_found_screen.dart';
import '../screens/patient_detail_screen.dart';

class AppRouter {
  static const String shell = '/';
  static const String patient = '/patient';

  static Route<dynamic> onGenerateRoute(RouteSettings settings) {
    switch (settings.name) {
      case shell:
        return MaterialPageRoute(
          builder: (_) => const AppShell(),
          settings: settings,
        );

      case patient:
        final args = (settings.arguments as Map?)?.cast<String, dynamic>() ?? {};
        return MaterialPageRoute(
          builder: (_) => PatientDetailScreen(
            patientId: args['patient_id']?.toString() ?? '',
            patientName: args['name']?.toString() ?? '',
            gender: args['gender']?.toString(),
          ),
          settings: settings,
        );

      default:
        return MaterialPageRoute(
          builder: (_) => const NotFoundScreen(),
          settings: settings,
        );
    }
  }
}