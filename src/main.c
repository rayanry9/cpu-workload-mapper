#include <stdio.h>
#include <string.h>
#include "temperature.h"
#include "temperature_grid.h"
#include "flp.h"
#include "util.h"

int main() {
  flp_t *flp1 = read_flp("fp_001_layer_0.flp", 0, 0);
  // flp_t *tim = read_flp(
  //     SOURCE_DIR
  //     "/Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001_tim.flp", 0, 0);

  thermal_config_t config = default_thermal_config();
  strcpy(config.model_type, "grid");
  strcpy(config.grid_layer_file, "fp_001.lcf");

  RC_model_t *model = alloc_RC_model(&config, flp1, 0, 0, 0, 0);
  populate_R_model(model, flp1);
  populate_C_model(model, flp1);

  double *temp = hotspot_vector(model);
  double *power = hotspot_vector(model);
  double *steady_temp = hotspot_vector(model);
  double *overall_power = hotspot_vector(model);

  set_temp(model, temp, model->config->init_temp);
  power[0] = 20.0;
  power[1] = 20.0;
  power[2] = 9.5;
  power[3] = 8.5;
  power[4] = 8.5;
  power[5] = 8.5;
  power[6] = 8.5;
  power[7] = 8.5;

  steady_state_temp(model, power, temp);

  for (int i = 0; i < model->grid->layers[0].flp->n_units; i++) {
    printf("Unit %s Temp: %f\n", model->grid->layers[0].flp->units[i].name,
           temp[i] - 273.15);
  }

  return 0;
}
