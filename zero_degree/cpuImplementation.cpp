#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <assert.h>
#include <math.h>

#include "dataFrame.hpp"


// R squared formula:
// - https://www.got-it.ai/solutions/excel-chat/excel-tutorial/r-squared/r-squared-in-excel

void linearRegressionConvolution(data_frame_desc_t *desc, float *out, long m) {
    // n must be odd
    //assert(m % 2 == 1);

    for(long i = 0; i < desc->numPoints; i++) {
        
        double sumXY = 0;
        double sumX = 0;
        double sumY = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;

        // Determine convoltion width n, which should be inversely preportional to r
        double r = sqrt(pow(desc->x[i], 2) + pow(desc->x[i], 2));
        long n = (long) (((double) m) * (20.0/r));

        // Don't ever require more than 500 points
        // TODO: Determine if there's a better number for this
        if(n > 500)
            n = 500;

        // Ensure n is odd
        if(n % 2 == 0)
            n += 1;
        
        for(long k = -n/2; k <= n/2; k++) {
            long convI = i + k;

            // Wrap around
            if(convI < 0) {
                convI = convI + desc->numPoints;
            } else if(convI > desc->numPoints - 1) {
                convI = convI - desc->numPoints;
            }

            sumXY += desc->x[convI] * desc->y[convI]; 
            sumX += desc->x[convI];
            sumY += desc->y[convI];
            sumXSquared += pow(desc->x[convI], 2);
            sumYSquared += pow(desc->y[convI], 2);
        }

        double r = (n*sumXY - sumX*sumY)
                    / sqrt(
                          (n*sumXSquared - pow(sumX, 2))
                        * (n*sumYSquared - pow(sumY, 2)));
        
        // Store r squared
        out[i] = (float) (pow(r, 2));
    }
}

int cpuPlaneExtract(data_frame_desc_t *desc) {
    float *convolutionOut = (float *) malloc(sizeof(float) * desc->numPoints);
    if(convolutionOut == nullptr) {
        fprintf(stderr, "Error: Couldn't malloc.\n");
        return -1;
    }

    linearRegressionConvolution(desc, convolutionOut, 7);

    // Next steps:
    //  1. Find the endpoints of each segment of points of length >= K which all have Rsquared >= W
    //  2. Comparing each segment with its neighbors, merge segments if their combined segments have a Rsquared >= W and their adjacent endpoints are within distance Q of one another
    //  3. Repeat step 2 until no more segments merge
    //  4. Print the endpoints of each segment in CSV (6 data values per line segment)
    //  5. (In Python) Plot each line segment together 

    // Print CSV
    for(uint32_t i = 0; i < desc->numPoints; i++) {
        printf("%f, %f, %f, %f\n", desc->x[i], desc->y[i], desc->z[i], convolutionOut[i]);
    }

    return 0;
}