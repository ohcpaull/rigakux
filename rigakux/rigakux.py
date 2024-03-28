import os #this deals with file directories
import io
import numpy as np
import re
from itertools import islice
import zipfile, io
import pandas as pd
import datetime
import imageio as imgio
from distutils.util import strtobool
import xml.etree.ElementTree as et
import pandas as pd


class RigakuAxis:
    """
    Axis information in Rigaku .rasx files
    """
    def __init__(self, parameters):
        
        self.name = parameters["Name"]
        self.unit = parameters["Unit"]
        self.offset = float(parameters["Offset"])
        self.position = float(parameters["Position"])
        self.state = parameters["State"]
        self.resolution = float(parameters["Resolution"])
    
    def __repr__(self):
        return f"RigakuAxis(name : {self.name}, position : {self.position}, state : {self.state})"
    
    def __str__(self):
        return ( 
            "## Axis parameters:\n"
            f"Name: {self.name}\n"
            f"Position : {self.position}\n"
            f"Mode : {self.state}\n"
        )
    
    def to_dict(self):
        return {
            'name' : self.name, 
            'unit' : self.unit, 
            'offset' : self.offset, 
            'position' : self.position, 
            'state' : self.state, 
            'resolution' : self.resolution
        }
    

class RigakuHardware:
    """
    Hardware information in Rigaku .rasx files
    """
    def __init__(self, parameters):
        self.goniometer = parameters["gonio"]["SelectedUnit"]
        self.attachment_head = parameters["head"]["SelectedUnit"]
        self.attachment_stage = parameters["stage"]["SelectedUnit"]
        self.detector = parameters["detector"]["SelectedUnit"]
        self.detector_mono = parameters["detector_mono"]["SelectedUnit"]
        self.receiving_atten = parameters["rec_atten"]["SelectedUnit"]

    def __repr__(self):
        return f"RigakuHardware( gonio : {self.goniometer}, head : {self.attachment_head}, detector : {self.detector}, atten : {self.receiving_atten})"

    def to_dict(self):
        return {
            'goniometer' : self.goniometer, 
            'attachment_head' : self.attachment_head, 
            'attachment_stage' : self.attachment_stage,
            'detector' : self.detector,
            'detector_mono' : self.detector_mono,
            'receiving_atten' : self.receiving_atten
        }


class RigakuFileRASX:
    """
    Represents a Rigaku .rasx X-ray diffraction file. There can 
    be multiple scans per file, and different scan axes in each
    scan. 

    Map files contain an image of the sample and a set of 
    sampling points, but the scans for each point need to be loaded
    separately using `RigakuFileRASX.load_map_scans`. Be warned that
    once this is done, the scans will not be in `RigakuFileRASX.scans`
    but in the `RigakuMapRASX.sampling_points` dictionary.

    Parameters
    ----------
    filename : str
        File name of the data file
    data_directory : str, os.path
        Directory where the file is stored

    Instance attributes
    -------------------
    file_path : os.path, BytesIO
        Full path to data file
    scan_metadata : list
        List of dictionaries for each scan within the file. Each
        dictionary contains data related to where different data
        is held within the .rasx file
    scans : list
        List of RigakuScanRASX objects that hold the scan data and 
        scan metadata information (such as axis positions and 
        hardware information)
    """
    def __init__(self, filename, data_directory="."):
        #print(type(filename))

        self.data_directory = data_directory
        self.scan_metadata = []
        if isinstance(filename, io.BytesIO):
            zipf = zipfile.ZipFile(filename)
            self.file_path = filename
        elif isinstance(filename, os.PathLike):
            zipf = zipfile.ZipFile(filename)
            self.file_path = filename
        elif isinstance(filename, str):
            self.file_path = os.path.join(data_directory, filename)
            zipf = zipfile.ZipFile(self.file_path)
            
        else:
            #zipf = zipfile.ZipFile(filename)
            raise TypeError(f"File is {type(filename)}")

        root = et.parse(io.BytesIO(zipf.read("root.xml"))).getroot()       
        res = root.find("Data0")
        if res is None:
            raise ValueError("No data inside rasx file!")
        

        # If Data0 type is SampleImage, make a map object
        if res.attrib["Type"] == "SampleImage":
            f = res.findall("ContentHashList")
            map_name = f[0].attrib["Name"]
            container = res.tag
            config_name = None
            self.map_metadata = (dict(
                container=container,
                scan_name=map_name,
                config_name=config_name
            ))
            self.map = RigakuMapRASX(self.file_path, self.map_metadata)


        # If Data0 type is Profile, data is either a single (or multiple) 
        # individual scans or a reciprocal space map
        elif res.attrib["Type"] == "Profile":
            self.scans = []
            self.scan_metadata = []
            i = 0
            while True:
                res = root.find(f"Data{i}")
                if res is None:
                    break
                f = res.findall("ContentHashList")
                scan_name = f[0].attrib["Name"]
                hardware_name = f[1].attrib["Name"]
                self.scan_metadata.append(dict(
                    container = res.tag,
                    scan_name = scan_name,
                    config_name = hardware_name
                ))
                self.scans.append(RigakuScanRASX(self.file_path, self.scan_metadata[i]))
                i += 1

            # If datatype is a reciprocal space map, take associated metadata
            if self.scans[0].metadata["data_type"] == 'RAS_3DE_RSM':
                self.rsm_metadata = self.scans[0].rsm_metadata
               
    def __str__(self):
        table = pd.DataFrame(columns=["Scan #", "Comment", "Scan Axis"])
        for idx, sc in enumerate(self.scans):
            comment = sc.metadata["comment"]
            scaxis = sc.scan_information["axis"]
            table.at[idx, "Scan #"] = idx
            table.at[idx, "Comment"] = comment
            table.at[idx, "Scan Axis"] = scaxis
        print(f"Rigaku RASX File: {os.path.basename(self.file_path)}")
              
        return table.to_markdown()

    def load_map_scans(self, data_directory="."):
        """
        If RigakuFileRASX is a XY map file, we need to load
        all the datasets for each XY point into the class.

        Important: This function assumes there is a folder
        which ONLY contains all the scan files in the map.

        TODO: Need to make this more general in case there 
        are multiple scans within a single .rasx file.

        Parameters
        ---------
        filestem : str
            The stem of the filename that will be iterated over.
            Naming convention is f"{filestem}_001.rasx
        data_directory : str or os.path
        """
        for idx, fname in enumerate(os.listdir(data_directory)):
            tmp = zipfile.ZipFile(os.path.join(data_directory, fname))
            root = et.parse(io.BytesIO(tmp.read("root.xml"))).getroot()

            res = root.find(f"Data0")
            if res is None:
                break
            f = res.findall("ContentHashList")
            scan_name = f[0].attrib["Name"]
            hardware_name = f[1].attrib["Name"]
            self.scan_metadata.append(dict(
                container = res.tag,
                scan_name = scan_name,
                config_name = hardware_name
            ))
            scan = RigakuScanRASX(os.path.join(data_directory, fname), self.scan_metadata[idx])
            x = scan.axes["X"].position
            y = scan.axes["Y"].position

            for point in self.map.sampling_points:
                if point["x_pos"] == x  and point["y_pos"] == y:
                    point["scan"] = scan
                    point["masked"] = False

    def plotkb(self, scan_num=0, scale="log", **kwargs):

        scan = self.scans[scan_num]
        x = scan.x()
        y = scan.y()
       
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(ylabel="Intensity (a.u.)", xlabel=f"{scan.scan_information['axis']}", yscale=scale, **kwargs)
        
        CuKalpha1_pos = 0
        CuKalpha2_pos = 0
        CuKbeta_pos = 0
        WLalpha1_pos = 0
        WLalpha2_pos = 0
        CuKalpha1_line = ax.axvline(CuKalpha1_pos, color="red")
        CuKalpha2_line = ax.axvline(CuKalpha2_pos, color="red", alpha=0.5)
        WLalpha1_line = ax.axvline(WLalpha1_pos, color="blue", alpha=0.5)
        WLalpha2_line = ax.axvline(WLalpha2_pos, color="blue", alpha=0.5)
        beta_line = ax.axvline(CuKbeta_pos, color="red", alpha=0.5)

        ymax = ax.get_ylim()
        text_CuKalpha1 = ax.text(0, ymax[1], r"K$\alpha$1")
        text_CuKalpha2 = ax.text(0, ymax[1], r"K$\alpha$2")
        text_CuKbeta = ax.text(0, ymax[1], r"K$\beta$")
        text_WLalpha1 = ax.text(0, ymax[1], r"WL$\alpha$1")
        text_WLalpha2 = ax.text(0, ymax[1], r"WL$\alpha$2")

        def on_move_x(event):
            CuK_alpha1 = 1.540593
            CuK_alpha2 = 1.544414
            CuK_beta = 1.392246
            # Tungsten L alpha lines from https://xdb.lbl.gov/Section1/Table_1-2.pdf
            WL_alpha1 = 1.476424
            WL_alpha2 = 1.487477

            position_x = event.xdata
            


            # theta = np.arcsin(lambda/2d)
            # d = lambda/2sin(theta)
            if isinstance(position_x, float):
                d = CuK_alpha1 / (2 * np.sin(np.radians(position_x/2)))
                x_CuKalpha2 = np.degrees(np.arcsin(CuK_alpha2/(2 * d)))
                x_beta = np.degrees(np.arcsin(CuK_beta/(2*d)))
                x_WLalpha1 = np.degrees(np.arcsin(WL_alpha1/(2*d)))
                x_WLalpha2 = np.degrees(np.arcsin(WL_alpha2/(2*d)))

                

                CuKalpha1_line.set_xdata(position_x)
                text_CuKalpha1.set_position((position_x, ymax[1]))

                CuKalpha2_line.set_xdata(x_CuKalpha2 * 2)
                text_CuKalpha2.set_position((x_CuKalpha2*2, ymax[1]*0.7))

                beta_line.set_xdata(x_beta * 2)
                text_CuKbeta.set_position((x_beta*2, ymax[1]))

                WLalpha1_line.set_xdata(x_WLalpha1*2)
                text_WLalpha1.set_position((x_WLalpha1*2, ymax[1]*0.7))

                WLalpha2_line.set_xdata(x_WLalpha2*2)
                text_WLalpha2.set_position((x_WLalpha2*2, ymax[1]))



            fig.canvas.draw() 
            fig.canvas.flush_events()


        
        plt.connect('motion_notify_event', on_move_x)
        plt.show()

        return fig, ax


class RigakuMapRASX:
    def __init__(self, filename, metadata):
        zipf = zipfile.ZipFile(filename)

        scan = zipf.read(f"{metadata['container']}/{metadata['scan_name']}")

        ns = {"rasx" : "http://www.w3.org/2001/XMLSchema"}
        tree = et.parse(io.BytesIO(scan))
        root = tree.getroot()

        self.map_information = self._get_map_metadata(root)
        self.sampling_points = self._get_sampling_points(root)
        self.image = imgio.imread(zipf.read(f"{metadata['container']}/Image0.png"))
        
    def _get_map_metadata(self, root):
        map_query = {
            "date" : ".//Date",
            "x_pixel_size" : ".//PixelSizeXmm",
            "y_pixel_size" : ".//PixelSizeYmm",
            "center_x" : ".//CenterX",
            "center_y" : ".//CenterY",
            "center_x_px" : ".//CenterXInPixels",
            "center_y_px" : ".//CenterYInPixels",
            "center_x_offset_px" : ".//CenterXOffsetInPixels",
            "center_y_offset_px" : ".//CenterYOffsetInPixels",
            "lens_ratio" : ".//LensRatio",
            "index" : ".//Index",
            "brightness" : ".//Brightness"
        }
        scan_info = {key : root.find(value).text for key, value in map_query.items()}

        scan_info["date"] = str(scan_info["date"])# , '%Y-%m-%dT%H:%M:%S.%f%z')
        scan_info["x_pixel_size"] = float(scan_info["x_pixel_size"])
        scan_info["y_pixel_size"] = float(scan_info["y_pixel_size"])
        scan_info["center_x"] = float(scan_info["center_x"])
        scan_info["center_y"] = float(scan_info["center_y"])
        scan_info["center_x_px"] = float(scan_info["center_x_px"])
        scan_info["center_y_px"] = float(scan_info["center_y_px"])
        scan_info["center_x_offset_px"] = float(scan_info["center_x_offset_px"])
        scan_info["center_y_offset_px"] = float(scan_info["center_y_offset_px"])
        scan_info["lens_ratio"] = str(scan_info["lens_ratio"])
        scan_info["index"] = int(scan_info["index"])
        scan_info["brightness"] = float(scan_info["brightness"])

        return scan_info

    def _get_sampling_points(self, root):
        d = []
        dtype = dict(
            x_pos = float,
            y_pos = float,
            z_pos = float,
            executed = strtobool,
            phi_pos = float,
            comment = str,
            order = int
        )
        sampling_points = root.find(".//SamplingPoints")

        sampling_query = {
            "x_pos" : ".//Xmm",
            "y_pos" : ".//Ymm",
            "z_pos" : ".//Zmm",
            "executed" : ".//IsExecute",
            "phi_pos" : ".//Phi",
            "comment" : ".//Comment",
            "order" : ".//OrderInList",
        }

        for idx, sample in enumerate(sampling_points):
            d.append({key : dtype[key](sample.find(value).text) for key, value in sampling_query.items()})
            d[idx]["masked"] = True

        return d
    
    def plot_over_image(self, key=None, ax=None, **kwargs):
        """
        Plots a contourf map over the image of the sample that has been mapped. 
        Once you have loaded your scan data into the map, you can analyse each 
        scan to obtain some sort of parameter (max intensity, peak position, etc.)
        through simple analysis and add it to the `sampling_points` list of dictionaries.

        This function will take the figure of merit from each dictionary and plot it
        on top of the image.

        Parameters
        ----------
        key : str
            Dictionary key of the figure of merit (e.g. "max_int")
        fig : matplotlib.pyplot.figure
            Optional figure to pass to place the plot
        **kwargs : dict
            extra keyword arguments for `contourf` plotting.

        Returns
        -------
        im : matplotlib.contour.QuadContourSet
            im returned from plot
        ax : matplotlib.axes
        """
        if not ax:
            fig, ax = plt.subplots()
        xpts = np.array([point["x_pos"] for point in self.sampling_points])
        ypts = np.array([point["y_pos"] for point in self.sampling_points])
        xmin = self.map_information['center_x'] - (self.map_information['center_x_px'] - self.map_information['center_x_offset_px']) * self.map_information['x_pixel_size'] 
        ymin = self.map_information['center_y'] - (self.map_information['center_y_px'] - self.map_information['center_y_offset_px']) * self.map_information['y_pixel_size'] 
        xmax = self.map_information['center_x'] + (self.map_information['center_x_px'] - self.map_information['center_x_offset_px']) * self.map_information['x_pixel_size'] 
        ymax = self.map_information['center_y'] + (self.map_information['center_y_px'] - self.map_information['center_y_offset_px']) * self.map_information['y_pixel_size'] 

        masked = np.ma.array([point["masked"] for point in self.sampling_points]).reshape(len(np.unique(ypts)), len(np.unique(xpts)))

        ax.imshow(self.image, extent=[xmin, xmax, ymin, ymax])
        ax.set(xlabel="X (mm)", ylabel="Y (mm)")

        if key:
            xpts_valid = np.array([point["x_pos"] for point in self.sampling_points if "max_int" in point.keys()])
            ypts_valid = np.array([point["y_pos"] for point in self.sampling_points if "max_int" in point.keys()])
            z = np.ma.array([point[key] for point in self.sampling_points]).reshape(len(np.unique(ypts)), len(np.unique(xpts)))
            self.masked_map = dict(x=np.unique(xpts_valid), y=np.unique(ypts_valid), z=z)
            im = ax.contourf(np.unique(xpts), np.unique(ypts), z, **kwargs)
        else:
            im = ax.scatter(xpts, ypts)
        return im, ax
     
        
class RigakuScanRASX:
    """
    Reads the new format of Rigaku XRD files - "RASX". These files are zipped
    and contain a .xml file with hardware-related information, as well as a 
    .txt file with the scan data. This class reads the hardware-related information
    to 

    Parameters
    ----------
    filename : str
        filename of the .rasx file
    data_directory : str or os.path, optional
        directory where the data is found

    Instance Attributes
    -------------------
    metadata : dict
        Information about the scan, such as:
        - sample_name              : Sample name (if provided in Rigaku software)
        - comment                  : Comment about the scan (useful for multi-scan files)
        - measurement_package_name : Measurement package name (type of XRD measurement process in software)
        - measurement_part_name    : Measurement part name (specific scan name, e.g. General Scan, )
        - data_type                : Type of data, i.e. reciprocal space map (RAS_3DE_RSM)

    scan_information : dict
        Information about the scan, such as:
            - axis           : str, Name of the scan axis
            - mode           : 
            - start_angle    : float, Angle of scan axis at start of measurement
            - end_angle      : float, Angle of scan axis at end of measurement
            - step           : float, Step angle between adjacent datapoints in measurement
            - resolution     : 
            - speed          : float, Speed of scan
            - speed_unit     : str, Units of scan speed (typically degrees/minute)
            - axis_unit      : str, Units of scan axis (typically degrees)
            - intensity_unit : str, Units of intensity (typically cts/sec)
            - start time     : datetime.datetime.strptime, Date and time at start of measurement
            - end_time       : datetime.datetime.strptime, Date and time at end of measurement
    
    hardware : RigakuHardware

    scan : dict
        scan axis : axis data for 

    """
    def __init__(self, filename, metadata):
        zipf = zipfile.ZipFile(filename)

        scan = zipf.read(f"{metadata['container']}/{metadata['scan_name']}")
        hardware = zipf.read(f"{metadata['container']}/{metadata['config_name']}")
        
        ns = {"rasx" : "http://www.w3.org/2001/XMLSchema"}
        tree = et.parse(io.BytesIO(hardware))
        root = tree.getroot()
        
        self.metadata = self._get_scan_metadata(root)
        if self.metadata["data_type"] == 'RAS_3DE_RSM':
            self.rsm_metadata = self._get_RSM_metadata(root)
        
        self.scan_information = self._get_scan_information(root)
        self.hardware = self._get_hardware_information(root)
        
        axesdict = self._get_axis_information(root)
        self.axes = {}
        for key in axesdict.keys():
            self.axes[key] = RigakuAxis(axesdict[key])
            
        data = pd.read_csv(io.BytesIO(scan), delimiter="\t", names=[self.scan_information["axis"], "intensity", "att"])
        self.scan = {
            self.scan_information["axis"] : data[self.scan_information["axis"]],
            "intensity" : data["intensity"],
            "attenuator" : data["att"]
        }

    def _get_axis_information(self, root):
        axis_query = {
            "Omega" : ".//*Axis[@Name='Omega']",
            "TwoTheta" :  ".//*Axis[@Name='TwoTheta']",
            "TwoThetaTheta" : ".//*Axis[@Name='TwoThetaTheta']",
            "TwoThetaOmega" : ".//*Axis[@Name='TwoThetaOmega']",
            "TwoThetaChi" : ".//*Axis[@Name='TwoThetaChi']",
            "Chi" : ".//*Axis[@Name='Chi']",
            "Phi" : ".//*Axis[@Name='Phi']",
            "Z" : ".//*Axis[@Name='Z']",
            "Alpha" : ".//*Axis[@Name='Alpha']",
            "Beta" : ".//*Axis[@Name='Beta']",
        }
        if self.hardware.attachment_head == "RxRy":
            axis_query.update({
            "Rx" : ".//*Axis[@Name='Rx']",
            "Ry" : ".//*Axis[@Name='Ry']"
            })
        if self.hardware.attachment_head == "XY-4inch":
            axis_query.update({
                "X" : ".//*Axis[@Name='X']",
                "Y" : ".//*Axis[@Name='Y']",
            })
        return {key : root.find(value).attrib for key, value in axis_query.items()}
    
    def _get_scan_information(self, root):
        scan_query = {
            "axis" : ".//*AxisName",
            "mode" : ".//*Mode",
            "start_angle" : ".//*Start",
            "end_angle" : ".//*Stop",
            "step" : ".//*Step",
            "resolution" : ".//Resolution",
            "speed" : ".//Speed",
            "speed_unit" : ".//SpeedUnit",
            "axis_unit" : ".//PositionUnit",
            "intensity_unit" : ".//IntensityUnit",
            "start_time" : ".//StartTime",
            "end_time" : ".//EndTime",
            "unequally_spaced" : ".//UnequalySpaced",
            "wavelength" : ".//WavelengthKalpha1"
        }
        scan_info = {key : root.find(value).text for key, value in scan_query.items()}

        scan_info["start_angle"] = float(scan_info["start_angle"])
        scan_info["end_angle"] = float(scan_info["end_angle"])
        scan_info["step"] = float(scan_info["step"])
        scan_info["resolution"] = float(scan_info["resolution"])
        scan_info["speed"] = float(scan_info["speed"])
        scan_info["start_time"] = datetime.datetime.strptime(scan_info["start_time"], '%Y-%m-%dT%H:%M:%SZ')
        scan_info["end_time"] = datetime.datetime.strptime(scan_info["end_time"], '%Y-%m-%dT%H:%M:%SZ')
        if scan_info["unequally_spaced"] == "True":
            scan_info["unequally_spaced"] = True
        else:
            scan_info["unequally_spaced"] = False
        scan_info["wavelength"] = float(scan_info["wavelength"])

        return scan_info

    def _get_hardware_information(self, root):
        hardware_query = {
            "gonio" : ".//*Category[@Name='Goniometer']",
            "head" :  ".//*Category[@Name='AttachmentHead']",
            "stage" : ".//*Category[@Name='AttachmentStage']",
            "detector" : ".//*Category[@Name='Detector']",
            "detector_mono" : ".//*Category[@Name='DetectorMonochromator']",
            "rec_atten" : ".//*Category[@Name='ReceivingAttenuator']",
            "incident_slit" : ".//*Axis[@Name='IS'][@Unit='mm']",
            "receiving_slit_1" : ".//*Axis[@Name='RS1'][@Unit='mm']",
            "receiving_slit_2" : ".//*Axis[@Name='RS2'][@Unit='mm']",
        }
        return RigakuHardware({key : root.find(value).attrib for key, value in hardware_query.items()})

    def _get_scan_metadata(self, root):
        metadata_query = {
            "sample_name" : "*.//SampleName",
            "comment" : "*.//Comment",
            "measurement_package_name" : "*.//PackageName",
            "measurement_part_name" : "*.//PartName",
            "data_type" : "*.//DataType",
        }
        return {key : root.find(value).text if root.find(value) is not None else None for key, value in metadata_query.items()}

    def _get_RSM_metadata(self, root):
        rsm_query = {
            "scan_relative" : ".//*ScanIsRelative",
            "step_relative" : ".//*StepIsRelative",
            "scan_axis_start" : ".//*ScanAxisStart",
            "scan_axis_stop" : ".//*ScanAxisStop",
            "scan_axis_step" : ".//*ScanAxisStep",
            "step_axis_name" : ".//*StepAxisName",
            "step_axis_start" : ".//*StepAxisStart",
            "step-axis_stop" : ".//*StepAxisStop",
            "step_axis_step" : ".//*StepAxisStep",
            "two_theta_origin" : ".//*TwoThetaOrigin",
            "omega_origin" : ".//*OmegaOrigin",
            "chi_origin" : ".//*ChiOrigin",
            "phi_origin" : ".//*PhiOrigin",
            "two_theta_chi_origin" : ".//*TwoThetaChiOrigin",
        }
        try:
            d = {key : root.find(value).text for key, value in rsm_query.items()}
        except:
            raise TypeError("Scan is not part of reciprocal space map!")
        
        # Convert true/false values to boolean
        if d["scan_relative"] == "false":
            d["scan_relative"] = False
        else: 
            d["scan_relative"] = True
        if d["step_relative"] == "false":
            d["step_relative"] = False
        else: 
            d["step_relative"] = True
        
        return d

    def x(self, xlim=None):
        x = self.scan[self.scan_information["axis"]]
        if not xlim:
            xlim=(-np.inf,np.inf)
        return x[x.between(xlim[0], xlim[1])]  #x.values[x.values>x.iloc[xlim[0]]:x.values<x.iloc[xlim[1]]]
    
    def y(self, xlim=None):
        x = self.scan[self.scan_information["axis"]]
        y = self.scan["intensity"]
        if not xlim:
            xlim=(-np.inf,np.inf)
        return y[x.between(xlim[0], xlim[1])]