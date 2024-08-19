# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import ants
from nilearn import image, masking
import nibabel as nib
import pandas as pd


in_dir = '/home/jason/Coding Folder/'
atlas = '/home/jason/matlab/spm8/canonical/avg152T1.nii'
out_dir = '/home/jason/centiloid_out/'


os.makedirs(out_dir)

#Skull stip
mni = ants.image_read(atlas)
#mni_stripped = brain_extraction(mni, modality='t1')

# Import gaain masks
ctx_voi = image.load_img(
	'/home/jason/Coding Folder/Centiloid_Std_VOI/Centiloid_Std_VOI/nifti/2mm/voi_ctx_2mm.nii'
	)

wcbm_voi = image.load_img(
	'/home/jason/Coding Folder/Centiloid_Std_VOI/Centiloid_Std_VOI/nifti/2mm/voi_WhlCbl_2mm.nii'
	)


#nib.save(atlas_ni, f'{out_dir}/atlasni.nii.gz')
#atlas=f'{out_dir}/atlasni.nii.gz'
fixed = mni

output_df = pd.DataFrame(columns=['ID', 'SUVR'])

for subjectpet, subjectmr in zip(sorted(
		os.listdir(f'{in_dir}YC-0_PET_5070/YC-0_PET_5070/nifti')), sorted(
			os.listdir(f'{in_dir}YC-0_MR/YC-0_MR/nifti'))):
	if subjectpet.startswith('.') or subjectmr.startswith('.'):
		# ignore hidden files and other junk
		continue

	subject_amyloid = f'{in_dir}YC-0_PET_5070/YC-0_PET_5070/nifti/{subjectpet}'
	subject_mr = f'{in_dir}YC-0_MR/YC-0_MR/nifti/{subjectmr}'

	print('Amyloid:', subject_amyloid)

	subject_out = f'{out_dir}{subjectpet}'

	os.makedirs(subject_out)

	# Get full file path to input images
	orig_file = subject_mr
	amyloid_file = subject_amyloid
	
	# Skull Strip Original T1 and mask
	raw = ants.image_read(orig_file)
	raw_pet = ants.image_read(amyloid_file)
	#mri_strip_mask = brain_extraction(raw, modality='t1')
	
	#print orientation of PET and MRI
	pet_img = nib.load(subject_amyloid)
	mr_img = nib.load(subject_mr)
	
	current_ornt_pet = nib.aff2axcodes(pet_img.affine)
	current_ornt_mri = nib.aff2axcodes(mr_img.affine)
	
	print(f'PET image orientation: {current_ornt_pet}')
	print(f'MRI image orientation: {current_ornt_mri}')
	
	#Apply mask with skull stripped
	#masked_image = ants.mask_image(raw, mri_strip_mask)

	# Load orig T1 image as moving image for registration
	moving = raw
	
	
	#warp PET to MR
	warp_pet = ants.registration(moving, raw_pet, type_of_transform='SyN')

	# Do Registration of Moving to Fixed
	reg = ants.registration(fixed, moving, type_of_transform='SyN')

	# Save warped orig
	warped_orig_file = f'{subject_out}/warped_orig.nii.gz'
	ants.image_write(reg['warpedmovout'], warped_orig_file)

	# Apply transform to amyloid image which is already in same space
	warped_amyloid_file = f'{subject_out}/warped_amyloid.nii.gz'
	amyloid = warp_pet['warpedmovout']
	warped_amyloid = ants.apply_transforms(fixed, amyloid, reg['fwdtransforms'])
	ants.image_write(warped_amyloid, warped_amyloid_file)
	
	#import image as nifti1
	xformed_PET = image.load_img(f'{subject_out}/warped_amyloid.nii.gz')
	
	resampled_PET = image.resample_to_img(xformed_PET, ctx_voi)
	
	#apply masks
	ctx_masked = masking.apply_mask(resampled_PET, ctx_voi)
	wcbm_masked = masking.apply_mask(resampled_PET, wcbm_voi)
	
	# Calculate the mean uptake in the VOIs
	mean_uptake_voi = ctx_masked.mean()
	mean_uptake_ref = wcbm_masked.mean()
	
	suvr = mean_uptake_voi/mean_uptake_ref
	
	row = [subjectpet, suvr]
	
	output_df.loc[len(output_df)] = row
	
output_df.to_csv(f'{out_dir}standard_centiloid_suvrs.csv', index=False)



