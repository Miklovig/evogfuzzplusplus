#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char* data;
} DerivationTree;

void buggy_function(DerivationTree* inp) {
    char* str = inp->data;
    int bigNumber = 1;

    if (strstr(str, "mem") != NULL) {
        // Simulate high memory usage
        int bigList = 0;
        for (int i = 0; i < 10000; i++) {
            bigList += i;
        }
    }
    if (strstr(str, "cpu") != NULL) {
        // Simulate high CPU usage
        int i = 0;
        while (i < 10000) {
            bigNumber *= i;
            i++;
        }
    }
    if (strstr(str, "time") != NULL) {
        // Simulate high time usage
        clock_t start_time = clock();
        while ((double)(clock() - start_time) / CLOCKS_PER_SEC < 0.1) {}
    }
    if (strstr(str, "cov") != NULL) {
        // Simulate more lines for coverage testing
        int doSomething = 1;
    }
    if (strstr(str, "error") != NULL) {
        // Simulate TypeError
        char* errorMsg = "TypeError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
        exit(1);
    }
    if (strstr(str, "exception") != NULL) {
        // Simulate ValueError
        char* errorMsg = "ValueError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
        exit(1);
    }
}
