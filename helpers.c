#include <stdio.h>
#include "helpers.h"
#include "constants.h"

void welcome(unsigned char master) {
    if (master) {
        printf("            .''       _            __\n");
        printf("  ._.-.___.' (`\\     (_)___  _____/ /_____  __  __\n");
        printf(" //(        ( `'    / / __ \\/ ___/ //_/ _ \\/ / / /\n");
        printf("'/ )\\ ).__. )      / / /_/ / /__/ ,< /  __/ /_/ /\n");
        printf("' <' `/ ._/'/   __/ /\\____/\\___/_/|_|\\___/\\__, /\n");
        printf("   ` /     /   /___/                     /____/ v%s\n\n", JCKY_VERSION);
    }
}

unsigned int round_up_multiple(unsigned int number, unsigned int multiple) {
    if (multiple == 0)
        return number;

    unsigned int remainder = number % multiple;
    if (remainder == 0)
        return number;

    return number + multiple - remainder;
}
