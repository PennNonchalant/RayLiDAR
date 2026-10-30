#include <raylidar/version.h>

#include <iostream>
#include <string_view>

int main(int argc, char *argv[])
{
    if (argc == 2 && std::string_view(argv[1]) == "--version")
    {
        std::cout << "RayLiDAR " << raylidar::GetVersion() << '\n';
        return 0;
    }

    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help"))
    {
        std::cout << "Usage: raylidar_cli [--help | --version]\n";
        return 0;
    }

    std::cerr << "Invalid arguments. Use --help for usage.\n";
    return 1;
}
