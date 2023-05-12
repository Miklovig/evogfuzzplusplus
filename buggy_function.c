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
        printf("Simulating high memory usage\n");
        // Simulate high memory usage
        int bigList = 0;
        for (int i = 0; i < 10000; i++) {
            bigList += i;
        }
    }
    if (strstr(str, "cpu") != NULL) {
        printf("Simulating high CPU usage\n");
        // Simulate high CPU usage
        int i = 0;
        while (i < 10000) {
            bigNumber *= i;
            i++;
        }
    }
    if (strstr(str, "time") != NULL) {
        printf("Simulating high time usage\n");
        // Simulate high time usage
        clock_t start_time = clock();
        while ((double)(clock() - start_time) / CLOCKS_PER_SEC < 0.1) {}
    }
    if (strstr(str, "cov") != NULL) {
        printf("Simulating more lines for coverage testing\n");
        // Simulate more lines for coverage testing
        int doSomething = 1;
    }
    if (strstr(str, "error") != NULL) {
        printf("Simulating TypeError\n");
        // Simulate TypeError
        char* errorMsg = "TypeError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
        // exit(1);
        return;
    }
    if (strstr(str, "exception") != NULL) {
        printf("Simulating ValueError\n");
        // Simulate ValueError
        char* errorMsg = "ValueError been thrown";
        int len = strlen(errorMsg);
        char* str_copy = (char*)malloc((len + 1) * sizeof(char));
        strcpy(str_copy, errorMsg);
        str_copy[len] = '\0';
        perror(str_copy);
        free(str_copy);
        // exit(1);
        return;
    }
}

#define MAX_INPUTS 100

int main(int argc, char* argv[]) {
    FILE* file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return 1;
    }

    char inputs[MAX_INPUTS][100];  // Assuming each input has a maximum length of 100 characters
    int numInputs = 0;

    // Read each line from the file and store it in the inputs array
    char line[100];
    while (fgets(line, sizeof(line), file) != NULL) {
        // Remove newline character at the end of the line
        if (line[strlen(line) - 1] == '\n') {
            line[strlen(line) - 1] = '\0';
        }

        // Copy the line to the inputs array
        strcpy(inputs[numInputs], line);
        numInputs++;
    }

    fclose(file);

    // Call the function for each input
    for (int i = 0; i < numInputs; i++) {
        // Call your function here using inputs[i]
        buggy_function(inputs[i]);
    }

    return 0;
}

