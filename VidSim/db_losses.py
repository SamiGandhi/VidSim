import network_losses_model as losses

DISTANCE = 100  # meters
FREQUENCY = 868e6  # Hz (868 MHz is generally more suitable for wildlife monitoring applications)
ENVIRONEMENT = "wildlife_zone"

#----------------------------------
HUMIDITY_LEVEL = 76.33  # percent
VEGETATION_DENSITY_LEVEL = 0.53  # arbitrary unit
snr,loss = losses.combined_loss_model(DISTANCE, FREQUENCY, ENVIRONEMENT, HUMIDITY_LEVEL, VEGETATION_DENSITY_LEVEL)
ber = losses.calculate_ber(snr)
print(f'Winter total losses {loss} db,  snr{snr} db, ber {ber}')

#--------------------------------



HUMIDITY_LEVEL = 72  # percent
VEGETATION_DENSITY_LEVEL = 0.54 
snr,loss = losses.combined_loss_model(DISTANCE, FREQUENCY, ENVIRONEMENT, HUMIDITY_LEVEL, VEGETATION_DENSITY_LEVEL)
ber = losses.calculate_ber(snr)
print(f'Spring total losses {loss} db,  snr{snr} db, ber {ber}')

#---------------------------------------


HUMIDITY_LEVEL = 57  # percent
VEGETATION_DENSITY_LEVEL = 0.37 
snr,loss = losses.combined_loss_model(DISTANCE, FREQUENCY, ENVIRONEMENT, HUMIDITY_LEVEL, VEGETATION_DENSITY_LEVEL)
ber = losses.calculate_ber(snr)
print(f'Summer total losses {loss} db,  snr{snr} db, ber {ber}')



#----------------------------------
HUMIDITY_LEVEL = 69  # percent
VEGETATION_DENSITY_LEVEL = 0.32  # arbitrary unit
snr,loss = losses.combined_loss_model(DISTANCE, FREQUENCY, ENVIRONEMENT, HUMIDITY_LEVEL, VEGETATION_DENSITY_LEVEL)
ber = losses.calculate_ber(snr)
print(f'Autumn total losses {loss} db,  snr{snr} db, ber {ber}')
#--------------------------------



'''
#----------------------------------
HUMIDITY_LEVEL = 50  # percent
VEGETATION_DENSITY_LEVEL = 1  # arbitrary unit
loss = losses.combined_loss_model(DISTANCE, FREQUENCY, ENVIRONEMENT, HUMIDITY_LEVEL, VEGETATION_DENSITY_LEVEL)
print(f'Humidity 50 ZONE Vegitation 1 {loss} db')
#--------------------------------
'''




