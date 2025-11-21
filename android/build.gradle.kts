import org.gradle.api.tasks.compile.JavaCompile
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.gradle.api.file.Directory

// Repositories for all modules
allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// --- Keep your custom build directory layout ---
val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()

rootProject.layout.buildDirectory.value(newBuildDir)

// Configure all subprojects
subprojects {
    // Put each subproject's build under the relocated build dir
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    layout.buildDirectory.value(newSubprojectBuildDir)

    // Make sure the :app module is evaluated first when others depend on it
    project.evaluationDependsOn(":app")

    // ✅ Force Java/Kotlin 17 everywhere (kills "source value 8 is obsolete")
    tasks.withType<JavaCompile>().configureEach {
        sourceCompatibility = "17"
        targetCompatibility = "17"
        // (Optional) hide the obsolete-options linter warning
        options.compilerArgs.add("-Xlint:-options")
    }
    tasks.withType<KotlinCompile>().configureEach {
        kotlinOptions.jvmTarget = "17"
    }
}

// Clean task respects the relocated build directory
tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}