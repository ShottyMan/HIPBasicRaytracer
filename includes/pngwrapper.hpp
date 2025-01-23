#ifndef PNG_WRAPPER_HPP
#define PNG_WRAPPER_HPP
#include "color.hpp"
#include <cstddef>
#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb/stb_image_write.h"
#include "common.hpp"


class PNG_Writter{

private:
    unsigned char* image;
    short channels;
    unsigned int image_height;
    unsigned int image_width;
    bool private_image = false;
public:

    PNG_Writter(unsigned char* image, short channels, unsigned int height, unsigned int width)
    {

        this->image = image;
        this->channels = channels;
        image_height = height;
        image_width = width;



    }

    ~PNG_Writter()
    {

        if(image != NULL)
        {
            delete[] this->image;
        }

    }

    PNG_Writter(color* image, short channels, unsigned int height, unsigned int width)
    {

        this->image = new unsigned char[height*width*channels];
        int array_counter = 0;
        for(int row_index = 0; row_index < height; row_index += 1)
        {

            
            for(int column_index = 0; column_index < width; column_index++){
                static const interval intensity(0.000, 0.999);
                this->image[array_counter] = 255 * intensity.clamp(linear_to_gamma(image[column_index + (row_index * width)].x));
                this->image[array_counter + 1] = 255 * intensity.clamp(linear_to_gamma(image[column_index + (row_index * width)].y));
                this->image[array_counter + 2] = 255 * intensity.clamp(linear_to_gamma(image[column_index + (row_index * width)].z));

                array_counter += 3;
            }        

        }

        this->channels = channels;
        image_width = width;
        image_height = height;

    }

    int create_png_file()
    {


        return stbi_write_png("test.png", this->image_width, this->image_height, this->channels, image, this->image_width * this->channels);

    }


};

#endif