/* Parlando.app main executable.
 *
 * macOS refuses to launch an app bundle whose CFBundleExecutable is a shell
 * script (LaunchServices error -10669 on macOS 26), so the bundle needs a
 * real Mach-O binary. This stub runs the generated launch script next to it
 * (Contents/Resources/launch.sh) as a CHILD process and waits for it,
 * forwarding termination signals.
 *
 * Why spawn instead of exec: when the LaunchServices-launched process itself
 * becomes the Python/AppKit process (exec keeps the PID), macOS 26 places
 * the NSStatusItem off-screen (position x=-1) and the menu bar icon never
 * shows; the same code run from a terminal is fine. A child process behaves
 * like a terminal-launched one, so the icon appears. Permissions still land
 * on Parlando: macOS attributes Microphone/Accessibility requests to the
 * *responsible* process, and a child's responsible process is this stub,
 * i.e. Parlando.app (exactly how a terminal's children are attributed to
 * the terminal).
 *
 * It contains no per-machine data, so the prebuilt binary shipped in the
 * package (src/parlando/assets/parlando-launcher) is identical everywhere.
 * Rebuild with scripts/build_launcher.sh.
 */
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static pid_t child = 0;

static void forward_signal(int sig) {
    if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
    char exe[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0) {
        fputs("parlando launcher: executable path too long\n", stderr);
        return 1;
    }
    char real[PATH_MAX];
    if (realpath(exe, real) == NULL) {
        perror("parlando launcher: realpath");
        return 1;
    }
    char script[PATH_MAX];
    /* dirname() is Contents/MacOS; the script lives in Contents/Resources. */
    snprintf(script, sizeof script, "%s/../Resources/launch.sh", dirname(real));

    char **args = calloc((size_t)argc + 2, sizeof *args);
    if (args == NULL) return 1;
    args[0] = "/bin/sh";
    args[1] = script;
    for (int i = 1; i < argc; i++) args[i + 1] = argv[i];

    if (posix_spawn(&child, "/bin/sh", NULL, NULL, args, environ) != 0) {
        perror("parlando launcher: spawn");
        return 1;
    }
    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    signal(SIGHUP, forward_signal);

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        /* interrupted by a forwarded signal: keep waiting for the child */
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
