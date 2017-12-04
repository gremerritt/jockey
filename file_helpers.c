#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "constants.h"
#include "file_helpers.h"
#include "neural_net.h"


char jcky_write_file(
    nn_type **data,
    nn_type **targets,
    const unsigned int records,
    const unsigned int data_len,
    const unsigned int targets_len,
    char *filename)
{
    FILE *file;
    char version[] = JCKY_VERSION;
    unsigned char type;
    unsigned int i;

    if (filename == NULL) {
        printf("No filename provided.");
        return 1;
    }

    // Parse the version number
    char *major_version_c = strtok(version, ".");
    char *minor_version_c = strtok(NULL, ".");
    char *patch_version_c = strtok(NULL, ".");
    const unsigned char major_version = (unsigned char)strtol(strtok(major_version_c, " "), NULL, 10);
    const unsigned char minor_version = (unsigned char)strtol(strtok(minor_version_c, " "), NULL, 10);
    const unsigned char patch_version = (unsigned char)strtol(strtok(patch_version_c, " "), NULL, 10);

    // Get the nn_type
    switch (sizeof(nn_type)) {
        case sizeof(float):
            type = (unsigned char)JCKY_FLOAT;
            break;
        case sizeof(double):
            type = (unsigned char)JCKY_DOUBLE;
            break;
        default:
            printf("Invalid nn_type. Must be one of: float, double.\n");
            return 1;
    }

    type <<= ((sizeof(unsigned char) * 8) / 2);
    file = fopen(filename, "wb");
    if (file != NULL) {
        fwrite("JCKY", sizeof(char), 4, file);
        fwrite(&type, sizeof(unsigned char), 1, file);
        fwrite(&major_version, sizeof(unsigned char), 1, file);
        fwrite(&minor_version, sizeof(unsigned char), 1, file);
        fwrite(&patch_version, sizeof(unsigned char), 1, file);
        fwrite(&data_len, sizeof(unsigned int), 1, file);
        fwrite(&targets_len, sizeof(unsigned int), 1, file);
        fwrite(&records, sizeof(unsigned int), 1, file);
        for (i=0; i<records; i++) {
            fwrite(data[i], sizeof(nn_type), data_len, file);
            fwrite(targets[i], sizeof(nn_type), targets_len, file);
        }
        fclose(file);
    }
    else {
        printf("Unable to write to data file.\n");
        return 1;
    }

    return 0;
}


void jcky_read_record(jcky_file *file, const unsigned int record, nn_type *batch, nn_type *targets) {
    const unsigned long int offset = file->offset + (file->bytes_per_record * record);
    fseek(file->stream, offset, SEEK_SET);
    fread(batch, file->datum_size, file->data_len, file->stream);
    fseek(file->stream, offset + file->bytes_per_data, SEEK_SET);
    fread(targets, file->datum_size, file->targets_len, file->stream);
}


jcky_file jcky_open_file(char *filename) {
    FILE *stream;
    char identifier[4];
    unsigned char type_byte, type;
    unsigned char major_version, minor_version, patch_version;
    unsigned int records, data_len, targets_len, bytes_per_record;
    unsigned long int expected_file_size;
    unsigned long int file_size;
    unsigned long int offset = (unsigned long int)jcky_file_byte_offset();
    jcky_file file;

    stream = fopen(filename, "rb");
    if (stream != NULL) {
        fread(identifier, sizeof(char), 4, stream);
        if (strncmp(identifier, "JCKY", 4) != 0) {
            printf("Error: %s is not a valid jockey file (missing identifier).\n", filename);
            fclose(stream);
        }
        else {
            fread(&type_byte, sizeof(unsigned char), 1, stream);
            type = type_byte >> ((sizeof(unsigned char) * 8) / 2);
            switch (type) {
                // Can maybe revist this at some point?
                // In theory we can just catch mismatches and cast appropriately.
                case (unsigned char)JCKY_FLOAT:
                    if (sizeof(nn_type) != sizeof(float)) {
                        printf("Error: %s has incorrect data type. File uses float, but Jockey is compiled with double.\n", filename);
                        fclose(stream);
                    }
                    bytes_per_record = sizeof(float);
                    break;
                case (unsigned char)JCKY_DOUBLE:
                    if (sizeof(nn_type) != sizeof(double)) {
                        printf("Error: %s has incorrect data type. File uses double, but Jockey is compiled with float.\n", filename);
                        fclose(stream);
                    }
                    bytes_per_record = sizeof(double);
                    break;
                default:
                    printf("Error: Invalid type identifier in %s.\n", filename);
                    fclose(stream);
            }

            if (stream != NULL) {
                fread(&major_version, sizeof(unsigned char), 1, stream);
                fread(&minor_version, sizeof(unsigned char), 1, stream);
                fread(&patch_version, sizeof(unsigned char), 1, stream);
                // Add any version checks here

                fread(&data_len, sizeof(unsigned int), 1, stream);
                fread(&targets_len, sizeof(unsigned int), 1, stream);
                fread(&records, sizeof(unsigned int), 1, stream);
                expected_file_size = (sizeof(nn_type) * records * (data_len + targets_len)) + offset;
                if (expected_file_size > LONG_MAX) {
                    printf("Warning: Unable verify correct file length for %s.\n", filename);
                }
                else {
                    fseek(stream, 0, SEEK_END);
                    file_size = ftell(stream);
                    if (file_size != expected_file_size) {
                        printf("Error: File size mismatch for %s.\n", filename);
                        fclose(stream);
                    }
                }
            }
        }
    }
    else {
        printf("Error: Unable to read %s.\n", filename);
    }

    file.stream = stream;
    file.datum_size = bytes_per_record;
    file.bytes_per_data = bytes_per_record * data_len;
    file.bytes_per_record = bytes_per_record * (data_len + targets_len);
    file.offset = (unsigned char)offset;
    file.data_len = data_len;
    file.targets_len = targets_len;

    return file;
}


char jcky_close_file(jcky_file file) {
    return (char)fclose(file.stream);
}


unsigned int jcky_get_num_inputs(jcky_file file) {
    unsigned int data_len;
    fseek(file.stream, sizeof(char) * 8, SEEK_SET);
    fread(&data_len, sizeof(unsigned int), 1, file.stream);
    return data_len;
}


unsigned int jcky_get_num_outputs(jcky_file file) {
    unsigned int targets_len;
    fseek(file.stream, (sizeof(char) * 8) + sizeof(unsigned int), SEEK_SET);
    fread(&targets_len, sizeof(unsigned int), 1, file.stream);
    return targets_len;
}


unsigned char jcky_file_byte_offset() {
    return (unsigned char)sizeof(char) * 8 +
           (unsigned char)sizeof(unsigned int) * 3;
}
