library("dplyr")
library("knitr")
library("kableExtra")
library("ggplot2")
library("sf")
library("rnaturalearth")
library("rnaturalearthdata")
library("tidyr")
library("ape")
library("patchwork")
library("GIFT")
sf::sf_use_s2(FALSE)

COUNTRY_LIST  <- c(
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", 
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", 
    "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", 
    "Liechtenstein", "Lithuania", "Luxembourg", "North Macedonia", "Malta", 
    "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", 
    "Portugal", "Romania", "San Marino", "Republic of Serbia", 
    "Slovakia", "Slovenia", "Spain", "Sweden", 
    "Switzerland", "Turkey", "Ukraine", "United Kingdom"
)


# Load the shape file using rnaturalearth
europe_shapefile <- ne_countries(scale = "medium", returnclass = "sf")

# First check column names in the shapefile
print(names(europe_shapefile))
# Then filter using the correct column names
europe_countries <- europe_shapefile %>%
    filter(name %in% COUNTRY_LIST | admin %in% COUNTRY_LIST)

# Create a single polygon that defines the study area by unioning all countries
study_area <- europe_countries %>%
    st_union() %>%
    st_as_sf()

# Add a buffer if needed (e.g., for coastal waters, set 0 if not needed)
# study_area_buffered <- st_buffer(study_area, dist = 0)
species_list <- GIFT_species()
write.csv(species_list, "data/GIFT/gift_species_list.csv", row.names = FALSE)

data <- GIFT_checklists(taxon_name = "Tracheophyta",
                         complete_taxon = TRUE,
                         complete_floristic = TRUE,
                         geo_type = "All",
                         shp = study_area,
                         overlap = "centroid_inside", 
                         remove_overlap = FALSE) # this argument adds two

# First the metadata of the checklists matching the different criteria, named
# $lists. The second element is a data.frame of all the checklists with the
# species composition per checklist ($checklists).
# Save the checklists data to a CSV file
write.csv(data$checklists, "data/GIFT/gift_checklists.csv", row.names = FALSE)

# Extract entity_ID values from the checklists data
checklist_entity_ids <- unique(data$checklists$entity_ID)

# Retrieve only the shapefiles for the entity_IDs in our checklists
gift_shapes <- GIFT_shapes(entity_ID = checklist_entity_ids)
# Filter gift_shapes to only include those that intersect with the study area
st_write(gift_shapes, "data/GIFT/gift_shapes.gpkg", driver = "GPKG", append = FALSE)

# # Filter gift_shapes to only include those that intersect with the study area
# gift_shapes <- gift_shapes %>%
#     st_transform(st_crs(study_area)) %>%  # Ensure both have the same CRS
#     filter(st_intersects(geometry, study_area, sparse = FALSE)[,1])


# st_write(gift_shapes, "gift_shapes.gpkg", driver = "GPKG", append = FALSE)
# st_write(gift_shapes, "gift_shapes_filtered.gpkg", driver = "GPKG", append = FALSE)




# rich_map <- dplyr::left_join(gift_shapes, data, by = "entity_ID") %>%
#   dplyr::filter(stats::complete.cases(total))

GIFT_species_lookup(genus = "Ammophila", epithet = "arenaria")
