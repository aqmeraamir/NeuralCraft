'''
Schematic Manager
-------------------

A module which allows for reading and creating minecraft .schematic files using python

Includes:
- create_schematic(filepath, dimensions, blocks, data): creates a .schematic file
- read_schematic(filepath): reads a .schematic file

-------------------

Author Github: https://github.com/aqmeraamir
'''

import nbtlib
from nbtlib import tag, parse_nbt



# Helper functions
def get_index_of_coordinates(dimensions, x, y, z):
    return (y * dimensions[2] + z) * dimensions[0] + x

# --------------------------------------
# Main functions to create/read schematic
# --------------------------------------

def create_schematic(filepath: str, blocks, data, dimensions: tuple):
    '''
    Creates a minecraft NBT .schematic file given some parameters

    Args:
        filepath (string): path to the .schematic file
        blocks (array): an array of all the blocks in the schematic
        data (array): other data
        dimensions (tuple): height, length, width (dimensions) of the schematic
    '''

    height, length, width = dimensions

    # Create the NBT structure
    schematic = nbtlib.tag.Compound({
        'Width': tag.Short(width),
        'Height': tag.Short(height),
        'Length': tag.Short(length),
        'Materials': tag.String('Alpha'),
        'Data': tag.ByteArray(data),
        'Blocks': tag.ByteArray(blocks),
    })

    # Create the NBT file
    nbt_file = nbtlib.File(schematic, root_name='Schematic', gzipped=True)
    nbt_file.save(filepath)

def read_schematic(filepath):
    '''
    Reads a minecraft NBT .schematic file

    Args:
        filepath (string): path to the .schematic file
    
    Returns:
        tuple: blocks, dimensions, data
    '''
    with nbtlib.load(filepath) as nbt_file:
        # Fetch dimensions
        width = nbt_file['Width']
        height = nbt_file['Height']
        length = nbt_file['Length']
        dimensions = height, length, width

        # Fetch other tags
        data = nbt_file['Data']
        blocks = nbt_file['Blocks']
        
        return blocks, dimensions, data


# --------------------------------------
# Test Usage
# --------------------------------------

# dimensions, blocks, data = read_schematic('data/schematics/1.schematic') # fetch a schematic

# blocks = list(blocks)
# blocks[get_index_of_coordinates(dimensions, 0, 1, 0)] = 14

# create_schematic("data/schematics/number.schematic", dimensions, blocks, data)
