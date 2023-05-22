#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char* data;
} DerivationTree;

void buggy_function(char* inp) {
    char* str = inp;
    int bigNumber = 1;

    if (strstr(str, "mem") != NULL) {
        //printf("Simulating high memory usage\n");
        // Simulate high memory usage
        int bigList = 0;
        for (int i = 0; i < 10000; i++) {
            bigList += i;
        }
    }
    if (strstr(str, "cpu") != NULL) {
        //printf("Simulating high CPU usage\n");
        // Simulate high CPU usage
        int i = 0;
        while (i < 10000) {
            bigNumber *= i;
            i++;
        }
    }
    if (strstr(str, "time") != NULL) {
        //printf("Simulating high time usage\n");
        // Simulate high time usage
        clock_t start_time = clock();
        while ((double)(clock() - start_time) / CLOCKS_PER_SEC < 0.1) {}
    }
    if (strstr(str, "cov") != NULL) {
        //printf("Simulating more lines for coverage testing\n");
        // Simulate more lines for coverage testing
        int doSomething = 1;
    }
    if (strstr(str, "error") != NULL) {
        //printf("Simulating TypeError\n");
        // Simulate TypeError
        char* errorMsg = "TypeError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
        exit(1);
        return;
    }
    if (strstr(str, "exception") != NULL) {
        //printf("Simulating ValueError\n");
        // Simulate ValueError
        char* errorMsg = "ValueError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
         exit(1);
        return;
    }
}

#define MAX_INPUTS 100

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        char* inp;
        inp = argv[i];
        //printf("inp came here %s", inp);
        buggy_function(inp);
    }
    return 0;
}

