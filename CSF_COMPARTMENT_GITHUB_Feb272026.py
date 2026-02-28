# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:26:44 2019

@author: atul
"""

import sys,argparse,inspect
from  Segmentation_Ventricle_Sulcus_CSF_1_Dec15_2019 import * 
def csf_compartments_ventbound_given(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z):
    returnvalue=0
    try:

        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_v1_with_vent_bound(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z)

        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def csf_compartments_ventobb_given(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z):
    returnvalue=0
    try:
        print("\n I AM AM AT ::{}".format(inspect.stack()[0][3]))
        subprocess.call("echo " + " AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        # sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb_ven_hem_given(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def csf_compartments_ventobb_no_hem(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z):
    returnvalue=0
    try:
        print("\n I AM AM AT ::{}".format(inspect.stack()[0][3]))
        subprocess.call("echo " + " AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        # sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb_ven_hem_given(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def csf_compartments_ventobb_no_hem_with_cistern(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z):
    returnvalue=0
    try:
        print("\n I AM AM AT ::{}".format(inspect.stack()[0][3]))
        subprocess.call("echo " + " AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb_with_cistern(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        # sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_with_vent_obb_ven_hem_given(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue

def call_csf_compartments_ventbound_no_hem(args):
    returnvalue=0
    try:
        subprocess.call("echo " + "SUCCEEDED 1 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        filename_gray=args.stuff[1]
        filename_mask=args.stuff[2]
        filename_bet=args.stuff[3]
        filename_ventricle_obb_mask=args.stuff[4]
        zoneV_min_z=int(args.stuff[5])
        zoneV_max_z=int(args.stuff[6])
        csf_compartments_ventobb_no_hem(filename_gray,filename_mask,filename_bet,filename_ventricle_obb_mask,zoneV_min_z,zoneV_max_z)
        subprocess.call("echo " + "SUCCEEDED 2 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        returnvalue=1
    except:
        subprocess.call("echo " + "FAILED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        pass
    return returnvalue

def call_csf_compartments_ventbound_no_hem_with_cis_1(args):
    returnvalue=0
    try:
        subprocess.call("echo " + "SUCCEEDED 1 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        filename_gray=args.stuff[1]
        filename_csf=args.stuff[2]
        filename_ventricle=args.stuff[3]
        filename_cistern=args.stuff[4]
        npyfiledirectory=args.stuff[5]
        # zoneV_min_z=int(args.stuff[5])
        # zoneV_max_z=int(args.stuff[6])
        process_csf_ventricle_cistern(filename_gray,filename_csf,filename_ventricle,filename_cistern,npyfiledirectory) ##,zoneV_min_z,zoneV_max_z)
        subprocess.call("echo " + "SUCCEEDED 2 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        returnvalue=1
    except:
        subprocess.call("echo " + "FAILED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        pass
    return returnvalue

def call_csf_compartments_ventbound_no_hem_with_cis(args):
    returnvalue=0
    try:
        subprocess.call("echo " + "SUCCEEDED 1 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        filename_gray=args.stuff[1]
        filename_mask=args.stuff[2]
        filename_bet=args.stuff[3]
        filename_ventricle_obb_mask=args.stuff[4]
        zoneV_min_z=int(args.stuff[5])
        zoneV_max_z=int(args.stuff[6])
        csf_compartments_ventobb_no_hem_with_cistern(filename_gray,filename_mask,filename_bet,filename_ventricle_obb_mask,zoneV_min_z,zoneV_max_z)
        subprocess.call("echo " + "SUCCEEDED 2 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        returnvalue=1
    except:
        subprocess.call("echo " + "FAILED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        pass
    return returnvalue


def csf_compartments(filename_gray,filename_mask,filename_bet):
    returnvalue=0
    try:

        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_v1(filename_gray,filename_mask,filename_bet)

        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def call_csf_compartments_ventbound_given(args):
    returnvalue=0
    try:
        filename_gray=args.stuff[1]
        filename_mask=args.stuff[2]
        filename_bet=args.stuff[3]
        zoneV_min_z=int(args.stuff[4])
        zoneV_max_z=int(args.stuff[5])
        csf_compartments_ventbound_given(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def call_csf_compartments_vent_obb_given(args):
    returnvalue=0
    try:
        subprocess.call("echo " + "SUCCEEDED 1 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        filename_gray=args.stuff[1]
        filename_mask=args.stuff[2]
        filename_bet=args.stuff[3]
        filename_ventricle_obb_mask=args.stuff[4]
        zoneV_min_z=int(args.stuff[5])
        zoneV_max_z=int(args.stuff[6])
        csf_compartments_ventobb_given(filename_gray,filename_mask,filename_bet,filename_ventricle_obb_mask,zoneV_min_z,zoneV_max_z)
        subprocess.call("echo " + "SUCCEEDED 2 AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        returnvalue=1
    except:
        subprocess.call("echo " + "FAILED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
        pass
    return returnvalue
def call_csf_compartments_ventbound_given_args():
    returnvalue=0
    print('I am at call_csf_compartments_ventbound_given_args' +sys.argv[1])
    try:
        print('I am at call_csf_compartments_ventbound_given_args')
        filename_gray=sys.argv[1]
        filename_mask=sys.argv[2]
        filename_bet=sys.argv[3]
        zoneV_min_z=int(sys.argv[4])
        zoneV_max_z=int(sys.argv[5])
        print('arguments::{}::{}::{}::{}::{}'.format(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z))
        csf_compartments_ventbound_given(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue

def call_csf_compartments(args):
    returnvalue=0


    try:
        filename_gray=args.stuff[1]
        filename_mask=args.stuff[2]
        filename_bet=args.stuff[3]
        csf_compartments(filename_gray,filename_mask,filename_bet)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def main():
    return_value=0
    print('I am before try at MAIN')
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('stuff', nargs='+')
        args = parser.parse_args()
        name_of_the_function=args.stuff[0]
        print('name of the function is ' + name_of_the_function)
        print("WO ZAI ::{}".format("main"))
        if name_of_the_function == "call_csf_compartments":
            print("WO ZAI ::{}".format("call_csf_compartments"))
            return_value=call_csf_compartments(args) #

        if name_of_the_function == "call_csf_compartments_ventbound_given":
            print("WO ZAI ::{}".format("call_csf_compartments_ventbound_given"))
            return_value=call_csf_compartments_ventbound_given(args)
        if name_of_the_function == "call_csf_compartments_vent_obb_given":
            print("WO ZAI ::{}".format("call_csf_compartments_vent_obb_given"))
            return_value=call_csf_compartments_vent_obb_given(args)
        if name_of_the_function == "call_csf_compartments_ventbound_no_hem":
            print("WO ZAI ::{}".format("call_csf_compartments_ventbound_no_hem"))
            return_value=call_csf_compartments_ventbound_no_hem(args)
        if name_of_the_function == "call_combine_sah_to_csf":
            print("WO ZAI ::{}".format("call_combine_sah_to_csf"))
            return_value=call_combine_sah_to_csf(args)
        if name_of_the_function == "call_csf_compartments_ventbound_no_hem_with_cis_1":
            print("WO ZAI ::{}".format("call_csf_compartments_ventbound_no_hem_with_cis_1"))
            return_value=call_csf_compartments_ventbound_no_hem_with_cis_1(args)

    except:
        x=0
        

    return return_value
def combine_sah_to_csf(sah_totalf,csf_maskf):

    # Load the binary masks
    sah_mask1 = nib.load(sah_totalf).get_fdata()
    csf_mask = nib.load(csf_maskf).get_fdata()

    # Create an empty mask with the same shape
    combined_mask = np.zeros_like(sah_mask1)

    # Assign unique labels
    combined_mask[sah_mask1 > 0.5] = 1  # Label 1 for first SAH mask
    combined_mask[csf_mask > 0.5] = 1   # Label 4 for CSF mask

    # Load header and affine from one of the original masks
    affine = nib.load(sah_totalf).affine
    header = nib.load(sah_totalf).header

    # Create a new NIfTI image
    combined_nifti = nib.Nifti1Image(combined_mask, affine, header)

    # Save the combined mask
    nib.save(combined_nifti, csf_maskf.split('.nii')[0]+'_with_sah.nii.gz')

    print("Combined mask saved as combined_mask.nii.gz")
def call_combine_sah_to_csf(args):
    returnvalue=0


    try:
        filename_csf=args.stuff[2]
        filename_sah=args.stuff[1]
        # filename_bet=args.stuff[3]
        combine_sah_to_csf(filename_sah,filename_csf) #,filename_bet)
        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue

if __name__ == '__main__':
    main()