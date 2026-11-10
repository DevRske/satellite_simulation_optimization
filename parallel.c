/* COMP.CE.350 Parallelization Excercise 2026
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi
                      Topi Leppanen  topi.leppanen@tuni.fi
                      Tapio Nevalainen tapio.nevalainen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
VERSION 24.0 - port to SDL2
VERSION 25.0 - add macOS support
*/


#ifdef _WIN32
#include "SDL.h"
#elif defined(__APPLE__)
#include "SDL.h"
#else
#include "SDL2/SDL.h"
#endif

#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

int mousePosX;
int mousePosY;

// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1920
#define SIZE WINDOW_WIDTH*WINDOW_HEIGHT

// The number of satellites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites (default: 64)
#define SATELLITE_COUNT 64

// These are used to control the satellite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000
#define BLACK_HOLE_RADIUS 4.5f


// Stores 2D data like the coordinates
typedef struct {
    float x;
    float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct {
    double x;
    double y;
} doublevector;

// Each float may vary from 0.0f ... 1.0f
typedef struct {
    float blue;
    float green;
    float red;
} color_f32;

// Stores rendered colors. Each value may vary from 0 ... 255
typedef struct {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
    uint8_t reserved;
} color_u8;

// Stores the satellite data, which fly around black hole in the space
typedef struct {
    color_f32 identifier;
    floatvector position;
    floatvector velocity;
} satellite;

// Pixel buffer which is rendered to the screen
color_u8 *pixels;

// Pixel buffer which is used for error checking
color_u8 *correctPixels;

// Buffer for all satellites in the space
satellite *satellites;
satellite *backupSatelites;


// ## You may add your own variables here ##
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#define CHECK(err, cause) if ((err) != CL_SUCCESS) { printf("%s failed: %d\n", cause, err); exit(1); }
#define LOCAL_GROUP_DIM_SIZE_X 8
#define LOCAL_GROUP_DIM_SIZE_Y 8
cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

cl_mem pixelsBuffer;
cl_mem satPosXBuffer;
cl_mem satPosYBuffer;
cl_mem satIdRedBuffer;
cl_mem satIdGreenBuffer;
cl_mem satIdBlueBuffer;

#define PHYSICS_THREAD_COUNT 6
#define BLOCK 16
#if SATELLITE_COUNT % BLOCK != 0
#error "SATELLITE_COUNT must be divisible by BLOCK"
#endif

// ## You may add your own initialization routines here ##
void init() {
    // Selecting the correct OpenCL Platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK(err, "platform")

    // Selecting a GPU device for OpenCL to use
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK(err, "device")
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    printf("%s\n", device_name);

    // Initializing the OpenCL context for the selected device
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK(err, "context")

    // Initializing the command queue for the device and context
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK(err, "queue")

    // Initializing the program object with the source code
    FILE *f = fopen("parallel.cl", "rb");
    if (f == NULL) { printf("Unable to open file\n"); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *source = malloc(size + 1);
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);
    const char *src = source;
    program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    free(source); // The source buffer is no longer required, as the program made a copy
    CHECK(err, "program")

    // Compile the program for the selected device
    char options[256];
    snprintf(options, sizeof(options),
        "-D WINDOW_WIDTH=%d -D SATELLITE_COUNT=%d -D SATELLITE_RADIUS=%ff -D BLACK_HOLE_RADIUS=%ff",
        WINDOW_WIDTH, SATELLITE_COUNT, SATELLITE_RADIUS, BLACK_HOLE_RADIUS);
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) { // Different error handling process in order to report build log in case of failed build
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *buildLog = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
        printf("%s\n", buildLog);
        free(buildLog);
        exit(1);
    }
    // Selection of the kernel from the compiled program
    kernel = clCreateKernel(program, "graphics", &err); // Placeholder name so far
    CHECK(err, "kernel")

    // Creation of buffers
    size_t bufferSize = SIZE * sizeof(color_u8);
    pixelsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &err);
    CHECK(err, "pixelsBuffer");
    bufferSize = SATELLITE_COUNT * sizeof(float);
    satPosXBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
    CHECK(err, "satPosXBuffer");
    satPosYBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
    CHECK(err, "satPosYBuffer");
    satIdRedBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
    CHECK(err, "satIdRedBuffer");
    satIdGreenBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
    CHECK(err, "satIdGreenBuffer");
    satIdBlueBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &err);
    CHECK(err, "satIdBlueBuffer");

    // These values do not change per frame, so they are enqueued only once
    float satelliteIdentifierRed[SATELLITE_COUNT];
    float satelliteIdentifierGreen[SATELLITE_COUNT];
    float satelliteIdentifierBlue[SATELLITE_COUNT];
    for (int j = 0; j < SATELLITE_COUNT; ++j) {
        satelliteIdentifierRed[j] = satellites[j].identifier.red;
        satelliteIdentifierGreen[j] = satellites[j].identifier.green;
        satelliteIdentifierBlue[j] = satellites[j].identifier.blue;
    }
    err = clEnqueueWriteBuffer(queue, satIdRedBuffer, CL_TRUE, 0,
    SATELLITE_COUNT * sizeof(float), satelliteIdentifierRed, 0, NULL, NULL);
    CHECK(err, "buffer satIdRed");
    err = clEnqueueWriteBuffer(queue, satIdGreenBuffer, CL_TRUE, 0,
        SATELLITE_COUNT * sizeof(float), satelliteIdentifierGreen, 0, NULL, NULL);
    CHECK(err, "buffer satIdGreen");
    err = clEnqueueWriteBuffer(queue, satIdBlueBuffer, CL_TRUE, 0,
        SATELLITE_COUNT * sizeof(float), satelliteIdentifierBlue, 0, NULL, NULL);
    CHECK(err, "buffer satIdBlue");

    // Setting up the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixelsBuffer);
    CHECK(err, "set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &satPosXBuffer);
    CHECK(err, "set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &satPosYBuffer);
    CHECK(err, "set arg 2");
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &satIdRedBuffer);
    CHECK(err, "set arg 3");
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &satIdGreenBuffer);
    CHECK(err, "set arg 4");
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &satIdBlueBuffer);
    CHECK(err, "set arg 5");
}

// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine)
// Moves the satellites based on gravity
// This is done multiple times in a frame because the Euler integration
// is not accurate enough to be done only once per frame.
void parallelPhysicsEngine() {
    int tmpMousePosX = mousePosX;
    int tmpMousePosY = mousePosY;

    // Double precision is required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    // [7.2] Replaced the previously used doublevector structure with one array per field (structure of arrays)
    // in order to have each field be in sequential memory locations and not require shuffling, given it is
    // accessed frequently during each Main Physics loop iteration. Physics frame time was reduced from 26ms to 17ms
    // using -O2 -ffast-math, while still passing the physics check.
    double tmpPosX[SATELLITE_COUNT];
    double tmpPosY[SATELLITE_COUNT];
    double tmpVelX[SATELLITE_COUNT];
    double tmpVelY[SATELLITE_COUNT];

    // Conversion of satellite state from float to double for increased precision.
    for (int idx = 0; idx < SATELLITE_COUNT; ++idx) { // [pp-2]
        tmpPosX[idx] = satellites[idx].position.x;
        tmpPosY[idx] = satellites[idx].position.y;
        tmpVelX[idx] = satellites[idx].velocity.x;
        tmpVelY[idx] = satellites[idx].velocity.y;
    }

    // [7.4] Implemented OpenMP Multi-Threading to the physics loop. Instead of the initial proposal of
    // performing a plain loop interchange (which did not give a positive improvement, 17ms to 20ms on -O2 -ffast-math),
    // all satellites are being divided into equally sized blocks (block size defined as BLOCK above in code) and
    // distributed among different threads running the same version of the Main Physics Iteration loop. The BlOCK size
    // was chosen after simple performance testing.
    #pragma omp parallel for num_threads(PHYSICS_THREAD_COUNT)
    for (int i = 0; i < SATELLITE_COUNT; i += BLOCK) {
        // [7.4] This is where the satellite positions are distributed into blocks. This is done in order to ensure
        // the compiler that the arrays do not overlap (there is no aliasing), since these arrays are declared inside
        // the parallel block, the compiler can be confident about that they don't alias, or that no-one modifies them
        // from the outside. Currently this is the expectation. The code was not tested without the new structure
        // of arrays.
        double px[BLOCK], py[BLOCK], vx[BLOCK], vy[BLOCK];
        for (int k = 0; k < BLOCK; ++k) {
            px[k] = tmpPosX[i + k];
            py[k] = tmpPosY[i + k];
            vx[k] = tmpVelX[i + k];
            vy[k] = tmpVelY[i + k];
        }

        // Main Physics Iteration loop - here, for every physics update (many per frame) a new state is calculated
        // for each satellite in the block. The outer loop in each iteration is responsible for calculating
        // one of new state per satellite in the block. Double precision is required here.
        for (int physicsUpdateIndex = 0; physicsUpdateIndex < PHYSICSUPDATESPERFRAME; ++physicsUpdateIndex) {
            // Physics Satellite Iteration loop - here, the next state of each satellite in this block is computed.
            for (int k = 0; k < BLOCK; ++k) {
                // Distance to the Black Hole
                // [7.2] Splitting the local doublevector temporary values (positionToBlackHole and normalizedDirection)
                // into seperate doubles gave no speedup as they are local temporary values (not arrays), so they stay
                // in registers and not get stored in memory as arrays are. This means that their layout does not matter.
                double positionToBlackHoleX = px[k] - tmpMousePosX;
                double positionToBlackHoleY = py[k] - tmpMousePosY;
                double distToBlackHoleSquared =
                        positionToBlackHoleX * positionToBlackHoleX +
                        positionToBlackHoleY * positionToBlackHoleY;
                double distToBlackHole = sqrt(distToBlackHoleSquared);

                // Gravity force calculation based on the distance to the Black Hole
                double normalizedDirectionX = positionToBlackHoleX / distToBlackHole;
                double normalizedDirectionY = positionToBlackHoleY / distToBlackHole;
                double accumulation = GRAVITY / distToBlackHoleSquared;

                // Update velocity based on force (Delta time is used to make velocity same despite different FPS)
                vx[k] -= accumulation * normalizedDirectionX * DELTATIME / PHYSICSUPDATESPERFRAME;
                vy[k] -= accumulation * normalizedDirectionY * DELTATIME / PHYSICSUPDATESPERFRAME;

                // Update position based on velocity
                px[k] += vx[k] * DELTATIME / PHYSICSUPDATESPERFRAME;
                py[k] += vy[k] * DELTATIME / PHYSICSUPDATESPERFRAME;
            }
        }

        for (int k = 0; k < BLOCK; ++k) {
            tmpPosX[i + k] = px[k];
            tmpPosY[i + k] = py[k];
            tmpVelX[i + k] = vx[k];
            tmpVelY[i + k] = vy[k];
        }
    }

    // Convert the final states back to floats and store them. [pp-1]
    for (int idx2 = 0; idx2 < SATELLITE_COUNT; ++idx2) {
        satellites[idx2].position.x = tmpPosX[idx2];
        satellites[idx2].position.y = tmpPosY[idx2];
        satellites[idx2].velocity.x = tmpVelX[idx2];
        satellites[idx2].velocity.y = tmpVelY[idx2];
    }
}

// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine)
// Decides the color for each pixel.
void parallelGraphicsEngine() {
    // Current mouse position in the window.
    int tmpMousePosX = mousePosX;
    int tmpMousePosY = mousePosY;

    // [7.2] Optimization loop which stores the required static values into sequential arrays, in order
    // to prevent repeated strided retrievals, which allows for easier auto-vectorization.

    float satellitePositionX[SATELLITE_COUNT];
    float satellitePositionY[SATELLITE_COUNT];
    for (int j = 0; j < SATELLITE_COUNT; ++j) {
        satellitePositionX[j] = satellites[j].position.x;
        satellitePositionY[j] = satellites[j].position.y;
    }

    // Allocating and writing to the buffers on the device
    err = clEnqueueWriteBuffer(queue, satPosXBuffer, CL_TRUE, 0,
        SATELLITE_COUNT * sizeof(float), satellitePositionX, 0, NULL, NULL);
    CHECK(err, "buffer satPosX");
    err = clEnqueueWriteBuffer(queue, satPosYBuffer, CL_TRUE, 0,
        SATELLITE_COUNT * sizeof(float), satellitePositionY, 0, NULL, NULL);
    CHECK(err, "buffer satPosY");

    // Setting additional args
    err = clSetKernelArg(kernel, 6, sizeof(int), &tmpMousePosX);
    CHECK(err, "set arg 6");
    err = clSetKernelArg(kernel, 7, sizeof(int), &tmpMousePosY);
    CHECK(err, "set arg 7");

    // Enqueuing kernel execution
    size_t global[2] = { WINDOW_WIDTH, WINDOW_HEIGHT };
    size_t local[2] = { LOCAL_GROUP_DIM_SIZE_X, LOCAL_GROUP_DIM_SIZE_Y };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local,
        0, NULL, NULL);
    CHECK(err, "enqueueNDRangeKernel");

    // Reading back the results
    err = clEnqueueReadBuffer(queue, pixelsBuffer, CL_TRUE, 0, SIZE * sizeof(color_u8), pixels,
        0, NULL, NULL);
    CHECK(err, "buffer pixels");
}

// ## You may add your own destrcution routines here ##
void destroy() {
    err = clFinish(queue);
    CHECK(err, "finish");
    err = clReleaseKernel(kernel);
    CHECK(err, "release kernel");
    err = clReleaseProgram(program);
    CHECK(err, "release program");
    err = clReleaseCommandQueue(queue);
    CHECK(err, "release command queue");
    err = clReleaseMemObject(pixelsBuffer);
    CHECK(err, "release buffer pixels");
    err = clReleaseMemObject(satPosXBuffer);
    CHECK(err, "release buffer satPosX");
    err = clReleaseMemObject(satPosYBuffer);
    CHECK(err, "release buffer satPosY");
    err = clReleaseMemObject(satIdRedBuffer);
    CHECK(err, "release buffer satIdRed");
    err = clReleaseMemObject(satIdGreenBuffer);
    CHECK(err, "release buffer satIdGreen");
    err = clReleaseMemObject(satIdBlueBuffer);
    CHECK(err, "release buffer satIdBlue");
    err = clReleaseContext(context);
    CHECK(err, "release context");
}


////////////////////////////////////////////////
// ¤¤ DO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
SDL_Window *win;
SDL_Surface *surf;
// Is used to find out frame times
int totalTimeAcc, satelliteMovementAcc, pixelColoringAcc, frameCount;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine() {
    // Graphics pixel loop
    for (int i = 0; i < SIZE; ++i) {
        // Row wise ordering
        floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

        // Draw the black hole
        floatvector positionToBlackHole = {
            .x = pixel.x -
                 HORIZONTAL_CENTER,
            .y = pixel.y - VERTICAL_CENTER
        };
        float distToBlackHoleSquared =
                positionToBlackHole.x * positionToBlackHole.x +
                positionToBlackHole.y * positionToBlackHole.y;
        float distToBlackHole = sqrt(distToBlackHoleSquared);
        if (distToBlackHole < BLACK_HOLE_RADIUS) {
            correctPixels[i].red = 0;
            correctPixels[i].green = 0;
            correctPixels[i].blue = 0;
            continue; // Black hole drawing done
        }

        // This color is used for coloring the pixel
        color_f32 renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

        // Find closest satellite
        float shortestDistance = INFINITY;

        float weights = 0.f;
        int hitsSatellite = 0;

        // First Graphics satellite loop: Find the closest satellite.
        for (int j = 0; j < SATELLITE_COUNT; ++j) {
            floatvector difference = {
                .x = pixel.x - satellites[j].position.x,
                .y = pixel.y - satellites[j].position.y
            };
            float distance = sqrt(difference.x * difference.x +
                                  difference.y * difference.y);

            if (distance < SATELLITE_RADIUS) {
                renderColor.red = 1.0f;
                renderColor.green = 1.0f;
                renderColor.blue = 1.0f;
                hitsSatellite = 1;
                break;
            } else {
                float weight = 1.0f / (distance * distance * distance * distance);
                weights += weight;
                if (distance < shortestDistance) {
                    shortestDistance = distance;
                    renderColor = satellites[j].identifier;
                }
            }
        }

        // Second graphics loop: Calculate the color based on distance to every satellite.
        if (!hitsSatellite) {
            for (int j = 0; j < SATELLITE_COUNT; ++j) {
                floatvector difference = {
                    .x = pixel.x - satellites[j].position.x,
                    .y = pixel.y - satellites[j].position.y
                };
                float dist2 = (difference.x * difference.x +
                               difference.y * difference.y);
                float weight = 1.0f / (dist2 * dist2);

                renderColor.red += (satellites[j].identifier.red *
                                    weight / weights) * 3.0f;

                renderColor.green += (satellites[j].identifier.green *
                                      weight / weights) * 3.0f;

                renderColor.blue += (satellites[j].identifier.blue *
                                     weight / weights) * 3.0f;
            }
        }
        correctPixels[i].red = (uint8_t) (renderColor.red * 255.0f);
        correctPixels[i].green = (uint8_t) (renderColor.green * 255.0f);
        correctPixels[i].blue = (uint8_t) (renderColor.blue * 255.0f);
    }
}

void sequentialPhysicsEngine(satellite *s) {
    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    doublevector tmpPosition[SATELLITE_COUNT];
    doublevector tmpVelocity[SATELLITE_COUNT];

    for (int i = 0; i < SATELLITE_COUNT; ++i) {
        tmpPosition[i].x = s[i].position.x;
        tmpPosition[i].y = s[i].position.y;
        tmpVelocity[i].x = s[i].velocity.x;
        tmpVelocity[i].y = s[i].velocity.y;
    }

    // Physics iteration loop
    for (int physicsUpdateIndex = 0;
         physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
         ++physicsUpdateIndex) {
        // Physics satellite loop
        for (int i = 0; i < SATELLITE_COUNT; ++i) {
            // Distance to the blackhole
            // (bit ugly code because C-struct cannot have member functions)
            doublevector positionToBlackHole = {
                .x = tmpPosition[i].x -
                     HORIZONTAL_CENTER,
                .y = tmpPosition[i].y - VERTICAL_CENTER
            };
            double distToBlackHoleSquared =
                    positionToBlackHole.x * positionToBlackHole.x +
                    positionToBlackHole.y * positionToBlackHole.y;
            double distToBlackHole = sqrt(distToBlackHoleSquared);

            // Gravity force
            doublevector normalizedDirection = {
                .x = positionToBlackHole.x / distToBlackHole,
                .y = positionToBlackHole.y / distToBlackHole
            };
            double accumulation = GRAVITY / distToBlackHoleSquared;

            // Delta time is used to make velocity same despite different FPS
            // Update velocity based on force
            tmpVelocity[i].x -= accumulation * normalizedDirection.x *
                    DELTATIME / PHYSICSUPDATESPERFRAME;
            tmpVelocity[i].y -= accumulation * normalizedDirection.y *
                    DELTATIME / PHYSICSUPDATESPERFRAME;

            // Update position based on velocity
            tmpPosition[i].x +=
                    tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
            tmpPosition[i].y +=
                    tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
        }
    }

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    // copy back the float storage.
    for (int i = 0; i < SATELLITE_COUNT; ++i) {
        s[i].position.x = tmpPosition[i].x;
        s[i].position.y = tmpPosition[i].y;
        s[i].velocity.x = tmpVelocity[i].x;
        s[i].velocity.y = tmpVelocity[i].y;
    }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_ERROR 10
#define ALLOWED_NUMBER_OF_ERRORS 10
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck() {
    int countErrors = 0;
    for (unsigned int i = 0; i < SIZE; ++i) {
        if (abs(correctPixels[i].red - pixels[i].red) > ALLOWED_ERROR ||
            abs(correctPixels[i].green - pixels[i].green) > ALLOWED_ERROR ||
            abs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_ERROR) {
            printf("Pixel x=%d y=%d value: %d, %d, %d. Should have been: %d, %d, %d\n",
                   i % WINDOW_WIDTH, i / WINDOW_WIDTH,
                   pixels[i].red, pixels[i].green, pixels[i].blue,
                   correctPixels[i].red, correctPixels[i].green, correctPixels[i].blue);
            countErrors++;
            if (countErrors > ALLOWED_NUMBER_OF_ERRORS) {
                printf("Too many errors (%d) in frame %d, Press enter to continue.\n", countErrors, frameNumber);
                getchar();
                return;
            }
        }
    }
    printf("Error check passed with acceptable number of wrong pixels: %d\n", countErrors);
}


// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void) {
    int timeSinceStart = SDL_GetTicks();

    // Error check during first frames
    if (frameNumber < 2) {
        memcpy(backupSatelites, satellites, sizeof(satellite) * SATELLITE_COUNT);
        sequentialPhysicsEngine(backupSatelites);
        mousePosX = HORIZONTAL_CENTER;
        mousePosY = VERTICAL_CENTER;
    } else {
        SDL_GetMouseState(&mousePosX, &mousePosY);
        if ((mousePosX == 0) && (mousePosY == 0)) {
            mousePosX = HORIZONTAL_CENTER;
            mousePosY = VERTICAL_CENTER;
        }
    }
    parallelPhysicsEngine();
    if (frameNumber < 2) {
        for (int i = 0; i < SATELLITE_COUNT; i++) {
            if (memcmp(&satellites[i], &backupSatelites[i], sizeof(satellite))) {
                printf("Incorrect satellite data of satellite: %d\n", i);
                getchar();
            }
        }
    }

    int satelliteMovementMoment = SDL_GetTicks();
    int satelliteMovementTime = satelliteMovementMoment - timeSinceStart;

    // Decides the colors for the pixels
    parallelGraphicsEngine();

    int pixelColoringMoment = SDL_GetTicks();
    int pixelColoringTime = pixelColoringMoment - satelliteMovementMoment;

    int finishTime = SDL_GetTicks();
    // Sequential code is used to check possible errors in the parallel version
    if (frameNumber < 2) {
        sequentialGraphicsEngine();
        errorCheck();
    } else if (frameNumber == 2) {
        previousFinishTime = finishTime;
        printf(
            "Time spent on moving satellites + Time spent on space coloring : Total time in milliseconds between frames (might not equal the sum of the left-hand expression)\n");
    } else if (frameNumber > 2) {
        // Print timings
        int totalTime = finishTime - previousFinishTime;
        previousFinishTime = finishTime;

        printf("Latency of this frame %i + %i : %ims \n",
               satelliteMovementTime, pixelColoringTime, totalTime);

        frameCount++;
        totalTimeAcc += totalTime;
        satelliteMovementAcc += satelliteMovementTime;
        pixelColoringAcc += pixelColoringTime;
        printf("Averaged over all frames: %i + %i : %ims.\n",
               satelliteMovementAcc / frameCount, pixelColoringAcc / frameCount, totalTimeAcc / frameCount);
    }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max) {
    return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed) {
    if (seed != 0) {
        srand(seed);
    }

    // Init pixel buffer which is rendered to the widow
    pixels = (color_u8 *) malloc(sizeof(color_u8) * SIZE);

    // Init pixel buffer which is used for error checking
    correctPixels = (color_u8 *) malloc(sizeof(color_u8) * SIZE);

    backupSatelites = (satellite *) malloc(sizeof(satellite) * SATELLITE_COUNT);


    // Init satellites buffer which are moving in the space
    satellites = (satellite *) malloc(sizeof(satellite) * SATELLITE_COUNT);

    // Create random satellites
    for (int i = 0; i < SATELLITE_COUNT; ++i) {
        // Random reddish color
        color_f32 id = {
            .red = randomNumber(0.f, 0.15f) + 0.1f,
            .green = randomNumber(0.f, 0.14f) + 0.0f,
            .blue = randomNumber(0.f, 0.16f) + 0.0f
        };

        // Random position with margins to borders
        floatvector initialPosition = {
            .x = HORIZONTAL_CENTER - randomNumber(50, 320),
            .y = VERTICAL_CENTER - randomNumber(50, 320)
        };
        initialPosition.x = (i / 2 % 2 == 0) ? initialPosition.x : WINDOW_WIDTH - initialPosition.x;
        initialPosition.y = (i < SATELLITE_COUNT / 2) ? initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

        // Randomize velocity tangential to the balck hole
        floatvector positionToBlackHole = {
            .x = initialPosition.x - HORIZONTAL_CENTER,
            .y = initialPosition.y - VERTICAL_CENTER
        };
        float distance = (0.06 + randomNumber(-0.01f, 0.01f)) /
                         sqrt(positionToBlackHole.x * positionToBlackHole.x +
                              positionToBlackHole.y * positionToBlackHole.y);
        floatvector initialVelocity = {
            .x = distance * -positionToBlackHole.y,
            .y = distance * positionToBlackHole.x
        };

        // Every other orbits clockwise
        if (i % 2 == 0) {
            initialVelocity.x = -initialVelocity.x;
            initialVelocity.y = -initialVelocity.y;
        }

        satellite tmpSatelite = {
            .identifier = id, .position = initialPosition,
            .velocity = initialVelocity
        };
        satellites[i] = tmpSatelite;
    }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void) {
    destroy();

    free(pixels);
    free(correctPixels);
    free(satellites);

    if (seed != 0) {
        printf("Used seed: %i\n", seed);
    }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window
void render(void) {
    SDL_LockSurface(surf);
    memcpy(surf->pixels, pixels, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
    SDL_UnlockSurface(surf);

    SDL_UpdateWindowSurface(win);
    frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits render window and starts mainloop
int main(int argc, char **argv) {
    if (argc > 1) {
        seed = atoi(argv[1]);
        printf("Using seed: %i\n", seed);
    }

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);
    win = SDL_CreateWindow(
        "Satellites",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        0
    );
    surf = SDL_GetWindowSurface(win);

    fixedInit(seed);
    init();

    SDL_Event event;
    int running = 1;
    while (running) {
        while (SDL_PollEvent(&event))
            switch (event.type) {
                case SDL_QUIT:
                    printf("Quit called\n");
                    running = 0;
                    break;
            }
        compute();
        render();
    }
    SDL_Quit();
    fixedDestroy();
}
