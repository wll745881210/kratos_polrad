############################################################
# The Target Binary Program
TARGET     := kratos

SRCDIR     := src
USRDIR     ?= 
EXTDIR     ?= 
INCDIR     :=
BUILDDIR   := obj
TARGETDIR  := bin
SRCEXT     := cpp
DEPEXT     := d
OBJEXT     := o
ARCH       ?= CUDA
DEBUG      ?= 0
MPI        ?= 0
LIB_EXT    ?= 

# Flags, Libraries and Includes
CFLAGS_CMD :=
INCDEP     :=
ifeq ($(ARCH),CUDA)
	CC     := nvcc
	ifeq ($(DEBUG),1)
	CFLAGS := -std=c++17 -rdc=true -arch=sm_80 -O0 -g \
              -include ./src/utilities/debug_macros.h \
	          -D __GPU_DEBUG__ -diag-suppress 3
	else   #DEBUG
	CFLAGS := -std=c++17 -rdc=true -arch=sm_80 -O3 \
              -include ./src/utilities/debug_macros.h \
			  $(CFLAGS_CMD)               
#			  --use_fast_math 
    endif  #DEBUG
	CFLAGS += -Wno-deprecated-gpu-targets
    INC    := -x cu
	LIB    := $(CFLAGS)
else ifeq ($(ARCH),HIPCPU)
	CC     := g++-10
	ifeq ($(DEBUG),1) 
	CFLAGS := -std=c++17 -O0 -g -fmax-errors=2 \
              -D __CPU_DEBUG__ \
              -include ./src/utilities/debug_macros.h
	else   #DEBUG
	CFLAGS := -std=c++17 -O3 $(CFLAGS_CMD) \
	 		  -fmax-errors=2 \
              -include ./src/utilities/debug_macros.h
	endif  #DEBUG
	LIB    := $(CFLAGS) -lpthread -ltbb
	INC    := -include hip/hip_runtime.h
else ifeq ($(ARCH),HIP)
	CC     := hipcc
	ifeq ($(DEBUG),1) 
	CFLAGS := -std=c++20 -O0 -g -include hip/hip_runtime.h \
	        -fgpu-rdc -fPIC
	else   #DEBUG
	CFLAGS := -std=c++17 -O3 -include hip/hip_runtime.h \
	        -fgpu-rdc -fPIC  -ffp-contract=fast \
			-DHIP_FAST_MATH # -ffast-math
	endif  #DEBUG
	CFLAGS += -Wno-return-type \
			  -Wno-unused-command-line-argument \
			  -Wno-inconsistent-missing-override
	LIB    := $(CFLAGS) --hip-link 
	INC    :=
else ifeq ($(ARCH),MUSA)
	CC     := mcc
	ifeq ($(DEBUG),1)
    CFLAGS := -std=c++17 -O2 -g $(CFLAGS_CMD)
	else   #DEBUG
	CFLAGS := -std=c++17 -Ofast -ffast-math $(CFLAGS_CMD) \
	         # --offload-arch=mp_21
    endif  #DEBUG
    INC    := -x musa
	LIB    := $(CFLAGS) -lmusa -lmusart -lpthread
endif

ifeq ($(MPI),1) 
CFLAGS     += -D__MPI__
LIB        += -lmpi
endif

ifeq ($(PRECISION),0)
CFLAGS     += -DPRECISION=0
else ifeq ($(PRECISION),2)
CFLAGS     += -DPRECISION=2
endif
LIB        += $(LIB_EXT)

# MY_VAR := $(shell echo $(USRDIR) | sed 's/\(.*\)\(\/\)\$/\1/' )

############################################################
# Automatic generating objs and deps

SOURCES := $(shell find -L $(SRCDIR) -type f \
             -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,\
            $(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

.PHONY : all dirs clean remake print

#Defauilt Make
all: dirs $(TARGET)

# print : ; $(info $$(SOURCES) is [${SOURCES}])
print : ; $(info $$(OBJECTS) is [${OBJECTS}])

# Remake
remake: clean all

# Make the Directories
dirs:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/$(USRDIR)
	@mkdir -p $(BUILDDIR)/$(USREXT)

#C lean only Objecst
clean:
	@$(RM) -rf $(BUILDDIR)/* $(TARGETDIR)/*

# Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))

# Link
$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGETDIR)/$(TARGET) $^ $(LIB)

# Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
	@$(CC) $(CFLAGS) $(INCDEP) -M $(SRCDIR)/$*.$(SRCEXT) >\
        $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) \
           $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' \
         < $(BUILDDIR)/$*.$(DEPEXT).tmp \
         > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' \
          < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | \
          sed -e 's/^ *//' -e 's/$$/:/' >> \
          $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

$(BUILDDIR)/$(USRDIR)/%.$(OBJEXT): $(USRDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
	@$(CC) $(CFLAGS) $(INCDEP) -M $(USRDIR)/$*.$(SRCEXT) >\
        $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT) \
           $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$(USRDIR)/$*.$(OBJEXT):|' \
         < $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp \
         > $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' \
        < $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp | fmt -1 |\
          sed -e 's/^ *//' -e 's/$$/:/' >> \
          $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp

$(BUILDDIR)/$(EXTDIR)/%.$(OBJEXT): $(EXTDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
	@$(CC) $(CFLAGS) $(INCDEP) -M $(EXTDIR)/$*.$(SRCEXT) >\
        $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT) \
           $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$(EXTDIR)/$*.$(OBJEXT):|' \
         < $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT).tmp \
         > $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' \
        < $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT).tmp | fmt -1 |\
          sed -e 's/^ *//' -e 's/$$/:/' >> \
          $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$(EXTDIR)/$*.$(DEPEXT).tmp
