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
    except:
        x=0
        

    return return_value


if __name__ == '__main__':
    main()