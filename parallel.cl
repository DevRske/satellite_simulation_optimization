__kernel void graphics(__global uchar4 *pixels,
                       __global const float *satellitePositionX,
                       __global const float *satellitePositionY,
                       __global const float *satelliteIdentifierRed,
                       __global const float *satelliteIdentifierGreen,
                       __global const float *satelliteIdentifierBlue,
                       const int tmpMousePosX,
                       const int tmpMousePosY)
{
        int pixel_x = get_global_id(0);
        int pixel_y = get_global_id(1);
        int i = pixel_y * WINDOW_WIDTH + pixel_x;

        // Draw the black hole
        float posToBlackHole_x = pixel_x - tmpMousePosX;
        float posToBlackHole_y = pixel_y - tmpMousePosY;
        float distToBlackHoleSquared = posToBlackHole_x * posToBlackHole_x + posToBlackHole_y * posToBlackHole_y;

        if (distToBlackHoleSquared < BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) {
            pixels[i].z = 0;
            pixels[i].y = 0;
            pixels[i].x = 0;
            return; // Black hole drawing done
        }

        // This color is used for coloring the pixel
        float renderColor_red = 0.f;
        float renderColor_green = 0.f;
        float renderColor_blue = 0.f;

        float squaredDistances[SATELLITE_COUNT];
        float computedWeights[SATELLITE_COUNT];
        float weights = 0.f;
        for (int j = 0; j < SATELLITE_COUNT; ++j) {
            float diff_x = pixel_x - satellitePositionX[j];
            float diff_y = pixel_y - satellitePositionY[j];

            float distanceSquared = (diff_x * diff_x + diff_y * diff_y);
            squaredDistances[j] = distanceSquared;
            computedWeights[j] = 1.0f / (distanceSquared * distanceSquared);
            weights += computedWeights[j];
        }

        // For finding the closest satellite to the current pixel.
        float shortestSquaredDistance = INFINITY;
        int closestSatelliteIndex = 0;

        for (int j = 0; j < SATELLITE_COUNT; ++j) {
            float squaredDistance = squaredDistances[j];

            int closerThanClosest = squaredDistance < shortestSquaredDistance;
            shortestSquaredDistance = closerThanClosest ? squaredDistance : shortestSquaredDistance;
            closestSatelliteIndex = closerThanClosest ? j : closestSatelliteIndex;
        }

        // If the pixel is within the satellite, it is painted white.
        if (shortestSquaredDistance < SATELLITE_RADIUS * SATELLITE_RADIUS) {
            renderColor_red = 1.0f;
            renderColor_green = 1.0f;
            renderColor_blue = 1.0f;
        } else // Else, the color is computed based on the satellites identifier.
        {
            renderColor_red = satelliteIdentifierRed[closestSatelliteIndex];
            renderColor_green = satelliteIdentifierGreen[closestSatelliteIndex];
            renderColor_blue = satelliteIdentifierBlue[closestSatelliteIndex];

            // Second graphics loop: Calculate the color based on the distance to every satellite. [pg-1]
            for (int k = 0; k < SATELLITE_COUNT; ++k) {
                float weight = computedWeights[k];

                renderColor_red += (satelliteIdentifierRed[k] * weight / weights) * 3.0f;
                renderColor_green += (satelliteIdentifierGreen[k] * weight / weights) * 3.0f;
                renderColor_blue += (satelliteIdentifierBlue[k] * weight / weights) * 3.0f;
            }
        }

        pixels[i].z = (uchar) (renderColor_red * 255.0f);
        pixels[i].y = (uchar) (renderColor_green * 255.0f);
        pixels[i].x = (uchar) (renderColor_blue * 255.0f);
}